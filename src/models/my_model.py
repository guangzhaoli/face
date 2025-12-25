import lightning as L
from diffusers.pipelines import FluxFillPipeline, FluxPriorReduxPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt
from PIL import Image
from .transformer import tranformer_forward
from .pipeline_tools import (
    encode_images,
    prepare_text_input,
    Flux_fill_encode_masks_images,
)
from .image_project import image_output
from .arcface_loss import DifferentiableFaceLoss


class InsertAnything(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        flux_redux_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe: FluxFillPipeline = (
            FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)
        )

        self.flux_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(flux_redux_id)
            .to(dtype=dtype)
            .to(device)
        )

        self.flux_redux.image_embedder.requires_grad_(False).eval()
        self.flux_redux.image_encoder.requires_grad_(False).eval()

        self.transformer = self.flux_fill_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        # Initialize face loss for face identity preservation
        self.use_face_loss = model_config.get("use_face_loss", False)
        self.use_multiscale_face_loss = model_config.get(
            "use_multiscale_face_loss", True
        )
        if self.use_face_loss:
            self.face_loss_weight = model_config.get("face_loss_weight", 0.1)
            arcface_weights = model_config.get("arcface_weights", None)
            arcface_model = model_config.get("arcface_model", "r50")  # r50 or r100
            arcface_root = model_config.get("arcface_root", None)
            self.face_align_mode = model_config.get("face_align_mode", "bbox")

            # Multi-scale layer weights (deeper = more identity-related)
            # Total effective weight ~2.0 (0.1+0.2+0.3+0.4+1.0)
            self.multiscale_layer_weights = model_config.get(
                "multiscale_layer_weights",
                {
                    "layer1": 0.1,  # Coarse face structure
                    "layer2": 0.2,  # Mid-level features
                    "layer3": 0.3,  # Fine facial details (eyes, nose, mouth)
                    "layer4": 0.4,  # High-level semantic features
                    "embedding": 1.0,  # Final identity embedding (most important)
                },
            )

            self.face_loss = DifferentiableFaceLoss(
                model_name=arcface_model,
                pretrained_path=arcface_weights,
                device=device,
                root=arcface_root,
                align_mode=self.face_align_mode,
            )
            loss_type = (
                "Multi-Scale" if self.use_multiscale_face_loss else "Single-Scale"
            )
            print(
                f"[InsertAnything] {loss_type} Face Loss enabled with weight {self.face_loss_weight}, align {self.face_align_mode}"
            )

        if self.use_face_loss:
            face_loss_temp = self.face_loss
            del self.face_loss
            self.to(device).to(dtype)
            self.face_loss = face_loss_temp.to(device=device, dtype=torch.float32)
        else:
            self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss, face_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        if face_loss is not None:
            self.log_face_loss = (
                face_loss.item()
                if not hasattr(self, "log_face_loss")
                else self.log_face_loss * 0.95 + face_loss.item() * 0.05
            )
            t_weight = (1 - self.last_t) ** 2
            self.log_t_weight = (
                t_weight
                if not hasattr(self, "log_t_weight")
                else self.log_t_weight * 0.95 + t_weight * 0.05
            )
        return step_loss

    def unpack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Unpack latents from transformer format to VAE format.

        This is the inverse of FluxPipeline._pack_latents().

        Official diffusers implementation:
        - _pack_latents: (B, C, H, W) -> (B, H/2, W/2, C, 2, 2) -> (B, N, C*4)
        - _unpack_latents: (B, N, C*4) -> (B, C, H, W)

        Args:
            latents: Packed latents from transformer, shape (B, N, C*4)
                     where N = (H/2) * (W/2), C*4 = 64 for Flux (16 channels * 4)
            height: Latent height (image_height // 8)
            width: Latent width (image_width // 8)

        Returns:
            Unpacked latents, shape (B, C, H, W) where C=16 for Flux
        """
        batch_size, num_patches, channels = latents.shape

        # The transformer output has condition channels concatenated
        # For Flux, image latents are 64 channels (16 * 4 for 2x2 packing)
        # Take only the first 64 channels if there are more
        if channels > 64:
            latents = latents[:, :, :64]
            channels = 64

        # Following official diffusers _unpack_latents logic:
        # latents: (B, N, C*4) where N = (H/2) * (W/2)
        # Reshape: (B, H/2, W/2, C, 2, 2)
        latents = latents.view(
            batch_size,
            height // 2,
            width // 2,
            channels // 4,  # This gives us 16 channels
            2,
            2,
        )

        # Permute: (B, H/2, W/2, C, 2, 2) -> (B, C, H/2, 2, W/2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        # Reshape: (B, C, H/2, 2, W/2, 2) -> (B, C, H, W)
        latents = latents.reshape(batch_size, channels // 4, height, width)

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using VAE.

        Args:
            latents: Latents in VAE format, shape (B, C, H, W)

        Returns:
            Decoded images, shape (B, 3, H*8, W*8), values in [0, 1]
        """
        vae = self.flux_fill_pipe.vae

        # Unscale latents
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor

        # Decode
        images = vae.decode(latents).sample

        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)

        return images

    def step(self, batch):
        imgs = batch["result"]
        src = batch["src"]
        mask = batch["mask"]
        ref = batch["ref"]
        data_type = batch.get("data_type", None)  # Get data type for face loss

        prompt_embeds = []
        pooled_prompt_embeds = []

        for i in range(ref.shape[0]):
            image_tensor = ref[i].cpu()

            image_tensor = image_tensor.permute(1, 2, 0)

            image_numpy = image_tensor.numpy()

            pil_image = Image.fromarray((image_numpy * 255).astype("uint8"))

            prompt_embed, pooled_prompt_embed = image_output(
                self.flux_redux, pil_image, self.device
            )

            prompt_embeds.append(prompt_embed.squeeze(1))

            pooled_prompt_embeds.append(pooled_prompt_embed.squeeze(1))

        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)

        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe,
            prompt_embeds=prompt_embeds.to(self.device),
            pooled_prompt_embeds=pooled_prompt_embeds.to(self.device),
        )

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_fill_pipe, imgs)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare conditions
            src_latents, mask_latents = Flux_fill_encode_masks_images(
                self.flux_fill_pipe, src, mask
            )

            condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

            # Save latent shape info for unpacking
            # NOTE: vae_scale_factor is for spatial downsampling (8 or 16)
            # vae.config.scaling_factor is for latent value scaling (~0.36)
            vae_scale_factor = self.flux_fill_pipe.vae_scale_factor
            latent_height = imgs.shape[2] // vae_scale_factor
            latent_width = imgs.shape[3] // vae_scale_factor

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            hidden_states=torch.cat((x_t, condition_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute MSE loss (velocity prediction)
        # target velocity = x_1 - x_0
        mse_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        # Compute face identity loss for person_head data type
        face_loss = None
        total_loss = mse_loss

        if self.use_face_loss and data_type is not None:
            # Check if any sample in batch is face-related data
            face_types = ("person_head", "person")
            is_face_batch = (
                any(dt in face_types for dt in data_type)
                if isinstance(data_type, (list, tuple))
                else data_type in face_types
            )

            if is_face_batch:
                try:
                    # ========== Differentiable Face Loss Computation ==========
                    #
                    # Flow Matching: x_t = (1-t) * x_0 + t * x_1
                    # Velocity: v = x_1 - x_0
                    # Therefore: x_0 = x_t - t * v = x_t - t * pred
                    #
                    # We estimate x_0 from the model's velocity prediction,
                    # then decode to pixel space and compute face loss.
                    # This allows gradients to flow back through the prediction.
                    # ===========================================================

                    # Estimate x_0 from velocity prediction (differentiable)
                    x_0_hat = x_t - t_ * pred

                    # Unpack latents from transformer format to VAE format
                    x_0_unpacked = self.unpack_latents(
                        x_0_hat, latent_height, latent_width
                    )

                    # Decode to pixel space (differentiable)
                    pred_images = self.decode_latents(x_0_unpacked)

                    # Extract target region (right half of diptych)
                    target_width = pred_images.shape[3] // 2
                    pred_target = pred_images[..., target_width:]  # Right half

                    # Compute differentiable face loss
                    # ref is the reference face image (ground truth)
                    # pred_target is the model's prediction (gradients flow through here)
                    # NOTE: Convert to float32 - numpy in detect_faces doesn't support bfloat16
                    if self.use_multiscale_face_loss:
                        # Multi-scale loss: combines features from all layers
                        face_loss = self.face_loss.forward_multiscale(
                            ref.float(),
                            pred_target.float(),
                            layer_weights=self.multiscale_layer_weights,
                        )
                    else:
                        # Single-scale loss: only uses final identity embedding
                        face_loss = self.face_loss(ref.float(), pred_target.float())

                    # T-aware weighting: face loss is more meaningful at low t
                    # t close to 0: x_0_hat ≈ x_0, high quality prediction
                    # t close to 1: x_0_hat is noisy, low quality prediction
                    # Use (1-t)^2 for smooth decay: t=0 -> 1.0, t=0.5 -> 0.25, t=1 -> 0
                    t_weight = ((1 - t.mean()) ** 2).clamp(0, 1)
                    effective_face_weight = self.face_loss_weight * t_weight

                    total_loss = mse_loss + effective_face_weight * face_loss

                except Exception as e:
                    # If face detection fails, just use MSE loss
                    print(f"[FaceLoss] Warning: {e}")
                    face_loss = None

        return total_loss, face_loss
