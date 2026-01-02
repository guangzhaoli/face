import lightning as L
from diffusers.pipelines import FluxFillPipeline, FluxPriorReduxPipeline
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt
from PIL import Image
import os
import cv2
import numpy as np
from .transformer import tranformer_forward
from .pipeline_tools import (
    encode_images,
    prepare_text_input,
    Flux_fill_encode_masks_images,
)
from .image_project import image_output
from .arcface_loss import DifferentiableFaceLoss

# Optional imports for perceptual losses
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[Warning] lpips not installed. Install with: pip install lpips")


# ============================================================================
# SSIM Loss Implementation (differentiable)
# ============================================================================


_ssim_kernel_cache: dict = {}


def gaussian_kernel(
    size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM computation (cached)."""
    cache_key = (size, sigma, channels, device, dtype)
    if cache_key in _ssim_kernel_cache:
        return _ssim_kernel_cache[cache_key]

    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g)
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    if device is not None or dtype is not None:
        kernel = kernel.to(device=device, dtype=dtype)

    _ssim_kernel_cache[cache_key] = kernel
    return kernel


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute differentiable SSIM loss.

    SSIM measures structural similarity considering luminance, contrast, and structure.
    Loss = 1 - SSIM (so lower is better, like other losses).

    Args:
        pred: Predicted images (B, C, H, W), values in [0, 1]
        target: Target images (B, C, H, W), values in [0, 1]
        window_size: Size of the Gaussian window
        sigma: Std of the Gaussian window
        reduction: 'mean', 'sum', or 'none'
        data_range: Range of valid values (1.0 for [0,1] images)

    Returns:
        SSIM loss (1 - SSIM), scalar or per-sample depending on reduction
    """
    _, channels, H, W = pred.shape

    min_dim = min(H, W)
    if window_size > min_dim:
        window_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        window_size = max(window_size, 3)

    kernel = gaussian_kernel(
        window_size, sigma, channels, device=pred.device, dtype=pred.dtype
    )

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pad = window_size // 2
    mu_pred = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred**2, kernel, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target**2, kernel, padding=pad, groups=channels) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_pred_target
    )

    sigma_pred_sq = sigma_pred_sq.clamp(min=0)
    sigma_target_sq = sigma_target_sq.clamp(min=0)

    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (
        sigma_pred_sq + sigma_target_sq + C2
    )
    ssim_map = numerator / denominator

    ssim_map = ssim_map.clamp(-1.0, 1.0)
    ssim_loss_map = 1.0 - ssim_map

    if reduction == "mean":
        return ssim_loss_map.mean()
    elif reduction == "sum":
        return ssim_loss_map.sum()
    else:
        return ssim_loss_map


def mask_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float = 2.0,
    background_weight: float = 1.0,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """
    Compute MSE loss with higher weight in masked (edited) regions.

    This encourages the model to pay more attention to the insertion area.

    Args:
        pred: Predicted latents/images (B, C, H, W) or (B, N, C)
        target: Target latents/images, same shape as pred
        mask: Binary mask (B, 1, H, W), 1 = edited region, 0 = background
        mask_weight: Weight for masked (edited) regions
        background_weight: Weight for background regions
        vae_scale_factor: VAE spatial downsampling factor (8 for Flux)

    Returns:
        Weighted MSE loss (normalized by weight sum)
    """
    mask = mask.to(device=pred.device, dtype=pred.dtype)

    mse = (pred - target) ** 2

    if mask.dim() == 4 and pred.dim() == 4:
        if mask.shape[2:] != pred.shape[2:]:
            mask = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
    elif mask.dim() == 4 and pred.dim() == 3:
        B, _, H_img, W_img = mask.shape
        H_latent = H_img // vae_scale_factor
        W_latent = W_img // vae_scale_factor

        mask_latent = F.interpolate(mask, size=(H_latent, W_latent), mode="nearest")

        N_packed = (H_latent // 2) * (W_latent // 2)
        if pred.shape[1] == N_packed:
            mask_packed = F.interpolate(
                mask_latent, size=(H_latent // 2, W_latent // 2), mode="nearest"
            )
            mask = mask_packed.view(B, -1, 1)
        else:
            mask = mask_latent.view(B, -1, 1)

    if mask.dim() == pred.dim() and mask.shape[-1] != mse.shape[-1]:
        if pred.dim() == 3:
            mask = mask.expand(-1, -1, mse.shape[-1])
        else:
            mask = mask.expand(-1, mse.shape[1], -1, -1)

    weights = mask * mask_weight + (1 - mask) * background_weight

    weighted_mse = (mse * weights).sum() / weights.sum()

    return weighted_mse


def debug_save_face_detection(
    ref: torch.Tensor,
    pred_target: torch.Tensor,
    t_value: float,
    step: int,
    face_loss_module,
    save_dir: str = "debug_face_detection",
):
    """
    保存人脸检测调试图像。

    Args:
        ref: 参考图像 (B, C, H, W), [0, 1]
        pred_target: 预测的目标图像 (B, C, H, W), [0, 1]
        t_value: 当前时间步 t
        step: 训练步数
        face_loss_module: DifferentiableFaceLoss 实例
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size = ref.shape[0]

    # 检测人脸
    with torch.no_grad():
        ref_faces = face_loss_module.detect_faces(ref.float())
        pred_faces = face_loss_module.detect_faces(pred_target.float())

    for i in range(min(batch_size, 2)):  # 只保存前2个样本
        # 转换为 numpy (H, W, C), [0, 255]
        ref_np = (ref[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_np = (pred_target[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

        # 转为 BGR (OpenCV 格式)
        ref_bgr = cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        # 在图像上绘制检测框
        ref_status = "NO_FACE"
        pred_status = "NO_FACE"

        if ref_faces[i] is not None:
            bbox = ref_faces[i]["bbox"]
            cv2.rectangle(
                ref_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
            )
            ref_status = "DETECTED"

        if pred_faces[i] is not None:
            bbox = pred_faces[i]["bbox"]
            cv2.rectangle(
                pred_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
            )
            pred_status = "DETECTED"

        # 添加文字标注
        cv2.putText(
            ref_bgr,
            f"REF: {ref_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            pred_bgr,
            f"PRED: {pred_status} | t={t_value:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 拼接图像
        combined = np.hstack([ref_bgr, pred_bgr])

        # 保存
        filename = f"step{step:06d}_batch{i}_t{t_value:.3f}_ref{ref_status}_pred{pred_status}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), combined)

    # 打印统计
    ref_detected = sum(1 for f in ref_faces if f is not None)
    pred_detected = sum(1 for f in pred_faces if f is not None)
    print(
        f"[DEBUG] Step {step}, t={t_value:.3f}: ref {ref_detected}/{batch_size}, pred {pred_detected}/{batch_size}"
    )


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
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

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

        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        self.lora_layers = self.init_lora(lora_path, lora_config)

        # ==================== Loss Configuration ====================
        self.use_t_aware_weighting = model_config.get("use_t_aware_weighting", True)

        # Face Identity Loss (ArcFace-based)
        self.use_face_loss = model_config.get("use_face_loss", False)
        self.use_multiscale_face_loss = model_config.get(
            "use_multiscale_face_loss", True
        )
        if self.use_face_loss:
            self.face_loss_weight = model_config.get("face_loss_weight", 0.1)
            arcface_weights = model_config.get("arcface_weights", None)
            arcface_model = model_config.get("arcface_model", "r50")
            arcface_root = model_config.get("arcface_root", None)
            self.face_align_mode = model_config.get("face_align_mode", "bbox")
            self.multiscale_layer_weights = model_config.get(
                "multiscale_layer_weights",
                {
                    "layer1": 0.1,
                    "layer2": 0.2,
                    "layer3": 0.3,
                    "layer4": 0.4,
                    "embedding": 1.0,
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
                f"[InsertAnything] {loss_type} Face Loss enabled: weight={self.face_loss_weight}, align={self.face_align_mode}"
            )

        # LPIPS Perceptual Loss
        self.use_lpips_loss = model_config.get("use_lpips_loss", False)
        self._lpips_model = None  # Will be set after to() call
        self._lpips_device = device
        if self.use_lpips_loss:
            if not LPIPS_AVAILABLE:
                raise ImportError(
                    "LPIPS loss requested but lpips not installed. Run: pip install lpips"
                )
            self.lpips_loss_weight = model_config.get("lpips_loss_weight", 0.1)
            self._lpips_net = model_config.get("lpips_net", "vgg")
            self._lpips_weights = model_config.get("lpips_weights", None)

            if self._lpips_weights is not None and not os.path.exists(
                self._lpips_weights
            ):
                raise FileNotFoundError(
                    f"LPIPS weights not found at '{self._lpips_weights}'. "
                    f"Download from: https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1"
                )

        # SSIM Structural Similarity Loss
        self.use_ssim_loss = model_config.get("use_ssim_loss", False)
        if self.use_ssim_loss:
            self.ssim_loss_weight = model_config.get("ssim_loss_weight", 0.1)
            self.ssim_window_size = model_config.get("ssim_window_size", 11)
            print(f"[InsertAnything] SSIM Loss enabled: weight={self.ssim_loss_weight}")

        # Mask-weighted MSE Loss (prioritize edited regions)
        self.use_mask_weighted_loss = model_config.get("use_mask_weighted_loss", False)
        if self.use_mask_weighted_loss:
            self.mask_region_weight = model_config.get("mask_region_weight", 2.0)
            self.background_region_weight = model_config.get(
                "background_region_weight", 1.0
            )
            print(
                f"[InsertAnything] Mask-weighted Loss enabled: mask={self.mask_region_weight}, bg={self.background_region_weight}"
            )

        # Move model to device (handle face_loss and lpips float32 separately)
        if self.use_face_loss:
            face_loss_temp = self.face_loss
            del self.face_loss
            self.to(device).to(dtype)
            self.face_loss = face_loss_temp.to(device=device, dtype=torch.float32)
        else:
            self.to(device).to(dtype)

        # Initialize LPIPS AFTER self.to() to keep it in float32
        if self.use_lpips_loss:
            import lpips

            self.lpips_model = lpips.LPIPS(
                net=self._lpips_net,
                model_path=self._lpips_weights,
            ).to(device=device, dtype=torch.float32)
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
            print(
                f"[InsertAnything] LPIPS Loss enabled: weight={self.lpips_loss_weight}, net={self._lpips_net}"
            )

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
        loss_dict = self.step(batch)
        total_loss = loss_dict["total_loss"]

        self.log_loss = (
            total_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + total_loss.item() * 0.05
        )

        if loss_dict.get("face_loss") is not None:
            face_loss_value = loss_dict["face_loss"].item()
            self.log_face_loss = (
                face_loss_value
                if not hasattr(self, "log_face_loss")
                else self.log_face_loss * 0.95 + face_loss_value * 0.05
            )
            t_weight = loss_dict.get("t_weight", 1.0)
            self.log_t_weight = (
                t_weight
                if not hasattr(self, "log_t_weight")
                else self.log_t_weight * 0.95 + t_weight * 0.05
            )
            self.log_face_loss_computed = True
        else:
            self.log_face_loss_computed = False

        if loss_dict.get("lpips_loss") is not None:
            lpips_value = loss_dict["lpips_loss"].item()
            self.log_lpips_loss = (
                lpips_value
                if not hasattr(self, "log_lpips_loss")
                else self.log_lpips_loss * 0.95 + lpips_value * 0.05
            )

        if loss_dict.get("ssim_loss") is not None:
            ssim_value = loss_dict["ssim_loss"].item()
            self.log_ssim_loss = (
                ssim_value
                if not hasattr(self, "log_ssim_loss")
                else self.log_ssim_loss * 0.95 + ssim_value * 0.05
            )

        return total_loss

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
        target_velocity = x_1 - x_0

        if self.use_mask_weighted_loss:
            mse_loss = mask_weighted_mse_loss(
                pred,
                target_velocity,
                mask,
                mask_weight=self.mask_region_weight,
                background_weight=self.background_region_weight,
                vae_scale_factor=vae_scale_factor,
            )
        else:
            mse_loss = F.mse_loss(pred, target_velocity, reduction="mean")

        self.last_t = t.mean().item()

        loss_dict = {
            "mse_loss": mse_loss,
            "face_loss": None,
            "lpips_loss": None,
            "ssim_loss": None,
            "t_weight": 1.0,
        }
        total_loss = mse_loss

        self.face_loss_skipped_no_data_type = False
        self.face_loss_skipped_not_face = False
        self.face_loss_detection_failed = False

        needs_pixel_decode = (
            self.use_face_loss or self.use_lpips_loss or self.use_ssim_loss
        )
        pred_images = None
        gt_images = None

        if needs_pixel_decode:
            t_latent = t_.to(dtype=x_t.dtype)
            x_0_hat = (x_t - t_latent * pred).to(dtype=x_t.dtype)
            x_0_unpacked = self.unpack_latents(x_0_hat, latent_height, latent_width)
            pred_images = self.decode_latents(x_0_unpacked)

            if self.use_lpips_loss or self.use_ssim_loss:
                with torch.no_grad():
                    x_0_gt_unpacked = self.unpack_latents(
                        x_0, latent_height, latent_width
                    )
                    gt_images = self.decode_latents(x_0_gt_unpacked)

        if self.use_t_aware_weighting:
            t_weight = (1 - t.mean()).clamp(min=0.1, max=1.0)
            loss_dict["t_weight"] = t_weight.item()
        else:
            t_weight = 1.0
            loss_dict["t_weight"] = 1.0

        if self.use_lpips_loss and pred_images is not None and gt_images is not None:
            lpips_input_pred = pred_images.float() * 2 - 1
            lpips_input_gt = gt_images.float() * 2 - 1
            lpips_val = self.lpips_model(lpips_input_pred, lpips_input_gt).mean()
            loss_dict["lpips_loss"] = lpips_val
            total_loss = total_loss + self.lpips_loss_weight * t_weight * lpips_val

        if self.use_ssim_loss and pred_images is not None and gt_images is not None:
            ssim_val = ssim_loss(
                pred_images.float(),
                gt_images.float(),
                window_size=self.ssim_window_size,
            )
            loss_dict["ssim_loss"] = ssim_val
            total_loss = total_loss + self.ssim_loss_weight * t_weight * ssim_val

        if self.use_face_loss:
            if data_type is None:
                self.face_loss_skipped_no_data_type = True
            else:
                face_types = ("person_head", "person")
                is_face_batch = (
                    any(dt in face_types for dt in data_type)
                    if isinstance(data_type, (list, tuple))
                    else data_type in face_types
                )

                if not is_face_batch:
                    self.face_loss_skipped_not_face = True
                elif pred_images is not None:
                    try:
                        target_width = pred_images.shape[3] // 2
                        pred_target = pred_images[..., target_width:]

                        if hasattr(self, "_debug_step_counter"):
                            self._debug_step_counter += 1
                        else:
                            self._debug_step_counter = 0

                        if self._debug_step_counter % 100 == 0:
                            debug_save_face_detection(
                                ref=ref,
                                pred_target=pred_target,
                                t_value=t.mean().item(),
                                step=self._debug_step_counter,
                                face_loss_module=self.face_loss,
                                save_dir="debug_face_detection",
                            )

                        if self.use_multiscale_face_loss:
                            face_loss_val, face_loss_details, valid_mask = (
                                self.face_loss.forward_multiscale(
                                    ref.float(),
                                    pred_target.float(),
                                    layer_weights=self.multiscale_layer_weights,
                                    return_details=True,
                                )
                            )
                            if not valid_mask.any():
                                self.face_loss_detection_failed = True
                                face_loss_val = None
                        else:
                            face_loss_val, _, valid_mask = self.face_loss(
                                ref.float(), pred_target.float(), return_details=True
                            )
                            if not valid_mask.any():
                                self.face_loss_detection_failed = True
                                face_loss_val = None

                        if face_loss_val is not None:
                            loss_dict["face_loss"] = face_loss_val
                            effective_face_weight = self.face_loss_weight * t_weight
                            total_loss = (
                                total_loss + effective_face_weight * face_loss_val
                            )

                    except Exception as e:
                        print(f"[FaceLoss] Warning: {e}")
                        self.face_loss_detection_failed = True

        loss_dict["total_loss"] = total_loss
        return loss_dict
