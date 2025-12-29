"""
ArcFace-based Face Identity Loss for LoRA Training

This module provides both differentiable and non-differentiable face identity loss:
- DifferentiableFaceLoss: Uses PyTorch ArcFace (IResNet) for gradient-based training
- ArcFaceLoss: Uses InsightFace ONNX for inference/evaluation only (no gradients)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from typing import Optional, Tuple


# ============================================================================
# IResNet Backbone (from insightface/arcface_torch)
# ============================================================================


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Basic residual block for IResNet."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("IBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """
    IResNet backbone for ArcFace.

    This is the PyTorch implementation from insightface/arcface_torch.
    Input: (B, 3, 112, 112) normalized face images
    Output: (B, 512) face embeddings
    """

    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers: list,
        dropout: float = 0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list] = None,
        fp16: bool = False,
    ):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False
    ):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

    def forward_multiscale(self, x: torch.Tensor) -> dict:
        """
        Extract multi-scale features from each layer.

        Returns a dict with:
        - 'layer1': (B, 64, 56, 56) - coarse features (face structure)
        - 'layer2': (B, 128, 28, 28) - mid-level features
        - 'layer3': (B, 256, 14, 14) - fine features (facial details)
        - 'layer4': (B, 512, 7, 7) - semantic features (identity)
        - 'embedding': (B, 512) - final identity embedding
        """
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        features["layer1"] = x  # (B, 64, 56, 56)

        x = self.layer2(x)
        features["layer2"] = x  # (B, 128, 28, 28)

        x = self.layer3(x)
        features["layer3"] = x  # (B, 256, 14, 14)

        x = self.layer4(x)
        features["layer4"] = x  # (B, 512, 7, 7)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        features["embedding"] = x  # (B, 512)

        return features


def iresnet50(**kwargs) -> IResNet:
    """IResNet-50 model."""
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)


def iresnet100(**kwargs) -> IResNet:
    """IResNet-100 model."""
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)


# ============================================================================
# Differentiable Face Loss (for training)
# ============================================================================


class DifferentiableFaceLoss(nn.Module):
    """
    Differentiable ArcFace-based identity loss for face generation training.

    Uses PyTorch IResNet backbone for face embedding extraction, allowing
    gradients to flow through for LoRA training optimization.

    Key features:
    - Fully differentiable (gradients flow through generated images)
    - Uses pretrained ArcFace weights from insightface
    - Face detection uses InsightFace ONNX (no grad needed for detection)
    - Face recognition uses PyTorch IResNet (differentiable)
    """

    def __init__(
        self,
        model_name: str = "r50",  # "r50" for IResNet-50, "r100" for IResNet-100
        pretrained_path: Optional[str] = None,
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
        root: Optional[str] = None,
        align_mode: str = "bbox",  # "bbox" or "kps"
    ):
        """
        Initialize differentiable face loss.

        Args:
            model_name: Model architecture ("r50" or "r100")
            pretrained_path: Path to pretrained weights (.pt or .pth file). REQUIRED.
                           Download from: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
                           Raises RuntimeError if file not found.
            device: Device to run on ("cuda", "cuda:0", "cpu", or torch.device)
            det_size: Face detection size (for InsightFace detector)
            root: Root directory for model storage
            align_mode: Face alignment mode ("bbox" or "kps")
        """
        super().__init__()
        self.device = device
        self.det_size = det_size
        self.root = root
        self.align_mode = self._normalize_align_mode(align_mode)

        # Initialize face recognition model (differentiable)
        if model_name == "r50":
            self.recognizer = iresnet50()
        elif model_name == "r100":
            self.recognizer = iresnet100()
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'r50' or 'r100'.")

        # Load pretrained weights if provided
        if pretrained_path is not None and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.recognizer.load_state_dict(state_dict)
            print(f"Loaded pretrained ArcFace weights from {pretrained_path}")
        else:
            raise RuntimeError(
                f"ArcFace weights not found at '{pretrained_path}'. "
                f"Face loss requires pretrained weights. "
                f"Download from: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch"
            )

        # IMPORTANT: Always keep recognizer in float32
        # ArcFace was trained in float32, and we need consistent dtype
        self.recognizer = self.recognizer.to(device=device, dtype=torch.float32)
        self.recognizer.eval()

        # Freeze recognizer weights (we only want gradients for input images)
        for param in self.recognizer.parameters():
            param.requires_grad = False

        # Face detector (non-differentiable, used only for face localization)
        self._detector = None
        self._detector_initialized = False

        # ArcFace input size
        self.face_size = 112

        # ArcFace 112x112 landmark template (InsightFace default)
        self.kps_template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        # Store expected dtype for input conversion
        self._model_dtype = torch.float32

    def _apply(self, fn):
        """
        Override _apply to keep recognizer in float32.

        When parent module calls .to(dtype=bfloat16), this ensures
        the recognizer weights stay in float32 for numerical stability.
        ArcFace was trained in float32 and works best in that precision.
        """
        # Apply function to all modules (handles device moves, etc.)
        super()._apply(fn)
        # Force recognizer back to float32 after any conversion
        self.recognizer = self.recognizer.to(dtype=torch.float32)
        # Keep internal device in sync after module moves
        try:
            self.device = next(self.recognizer.parameters()).device
        except StopIteration:
            pass
        return self

    def _parse_device_id(self) -> int:
        """Parse GPU device index from device string or torch.device."""
        device = self.device

        # Handle torch.device objects
        if isinstance(device, torch.device):
            if device.type == "cuda":
                return device.index if device.index is not None else 0
            return -1  # CPU

        # Handle string devices
        if isinstance(device, str):
            if device == "cpu":
                return -1
            if device == "cuda" or device == "cuda:0":
                return 0
            if device.startswith("cuda:"):
                try:
                    return int(device.split(":")[1])
                except (ValueError, IndexError):
                    return 0

        # Handle int (direct GPU index)
        if isinstance(device, int):
            return device if device >= 0 else -1

        return -1  # Default to CPU

    def _init_detector(self):
        """Lazy initialization of face detector."""
        if self._detector_initialized:
            return

        try:
            from insightface.app import FaceAnalysis

            fa_kwargs = {
                "name": "buffalo_l",
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            }
            if self.root is not None:
                fa_kwargs["root"] = self.root
            self._detector = FaceAnalysis(**fa_kwargs)
            ctx_id = self._parse_device_id()
            self._detector.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self._detector_initialized = True
        except ImportError:
            print("Warning: InsightFace not available. Face detection will not work.")
            self._detector_initialized = True

    def _normalize_align_mode(self, align_mode: str) -> str:
        if align_mode not in ("bbox", "kps"):
            print(
                f"Warning: Unknown align_mode '{align_mode}', falling back to 'bbox'."
            )
            return "bbox"
        return align_mode

    @torch.no_grad()
    def detect_faces(self, images: torch.Tensor) -> list:
        """
        Detect faces in images and return face info.

        Args:
            images: (B, C, H, W) tensor, values in [0, 1]

        Returns:
            List of face info dicts {"bbox": np.ndarray, "kps": np.ndarray},
            or None if no face detected
        """
        self._init_detector()
        if self._detector is None:
            return [None] * images.shape[0]

        batch_faces = []
        for i in range(images.shape[0]):
            img = images[i].detach()
            # Convert to numpy BGR
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            faces = self._detector.get(img_np)
            if len(faces) > 0:
                # Get the most confident face
                face = max(faces, key=lambda x: x.det_score)
                bbox = face.bbox.astype(int)
                kps = getattr(face, "kps", None)
                if kps is not None:
                    kps = kps.astype(np.float32)
                    if kps.shape != (5, 2):
                        kps = None
                batch_faces.append({"bbox": bbox, "kps": kps})
            else:
                batch_faces.append(None)

        return batch_faces

    def crop_and_resize_face(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
    ) -> torch.Tensor:
        """
        Differentiably crop and resize face region.

        Args:
            image: (C, H, W) tensor
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            (C, 112, 112) cropped face tensor
        """
        C, H, W = image.shape
        x1, y1, x2, y2 = bbox

        # Clamp bbox to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Expand bbox slightly for better face coverage
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(w, h) * 1.2

        x1, y1 = int(cx - size / 2), int(cy - size / 2)
        x2, y2 = int(cx + size / 2), int(cy + size / 2)

        # Pad if needed
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Crop face region
        face = image[:, y1:y2, x1:x2]

        # Pad if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            face = F.pad(
                face, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
            )

        # Resize to 112x112 using bilinear interpolation (differentiable)
        face = face.unsqueeze(0)  # (1, C, H, W)
        face = F.interpolate(
            face,
            size=(self.face_size, self.face_size),
            mode="bilinear",
            align_corners=False,
        )
        face = face.squeeze(0)  # (C, 112, 112)

        return face

    def _affine_to_theta(
        self,
        affine: torch.Tensor,
        in_size: Tuple[int, int],
        out_size: Tuple[int, int],
        align_corners: bool = False,
    ) -> torch.Tensor:
        """Convert pixel-space affine to normalized affine for grid_sample."""
        in_h, in_w = in_size
        out_h, out_w = out_size
        device = affine.device
        dtype = affine.dtype

        affine_3x3 = torch.eye(3, device=device, dtype=dtype)
        affine_3x3[:2, :] = affine

        if align_corners:
            t_out = torch.tensor(
                [
                    [(out_w - 1) / 2, 0.0, (out_w - 1) / 2],
                    [0.0, (out_h - 1) / 2, (out_h - 1) / 2],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
            t_in = torch.tensor(
                [
                    [2.0 / (in_w - 1), 0.0, -1.0],
                    [0.0, 2.0 / (in_h - 1), -1.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
        else:
            t_out = torch.tensor(
                [
                    [out_w / 2.0, 0.0, (out_w - 1) / 2.0],
                    [0.0, out_h / 2.0, (out_h - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
            t_in = torch.tensor(
                [
                    [2.0 / in_w, 0.0, (1.0 / in_w) - 1.0],
                    [0.0, 2.0 / in_h, (1.0 / in_h) - 1.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )

        theta = t_in @ affine_3x3 @ t_out
        return theta[:2, :]

    def _warp_affine(
        self,
        image: torch.Tensor,
        affine: np.ndarray,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Warp image with affine (pixel space) using grid_sample."""
        out_h, out_w = out_size
        c, h, w = image.shape
        device = image.device
        dtype = image.dtype

        affine_t = torch.tensor(affine, device=device, dtype=dtype)
        theta = self._affine_to_theta(
            affine_t, (h, w), (out_h, out_w), align_corners=False
        )
        grid = F.affine_grid(
            theta.unsqueeze(0),
            [1, c, out_h, out_w],
            align_corners=False,
        )
        warped = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped.squeeze(0)

    def align_face_by_kps(
        self,
        image: torch.Tensor,
        kps: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Align face using keypoints and similarity transform."""
        if kps is None or kps.shape != (5, 2):
            return None

        estimate = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), self.kps_template, method=cv2.LMEDS
        )
        if estimate is None:
            return None

        affine = estimate[0]
        if affine is None:
            return None

        affine_inv = cv2.invertAffineTransform(affine)
        return self._warp_affine(image, affine_inv, (self.face_size, self.face_size))

    def extract_face(
        self,
        image: torch.Tensor,
        face_info: Optional[dict],
    ) -> Optional[torch.Tensor]:
        """Extract aligned/cropped face based on current align_mode."""
        if face_info is None:
            return None

        if self.align_mode == "kps":
            aligned = self.align_face_by_kps(image, face_info.get("kps"))
            if aligned is not None:
                return aligned

        bbox = face_info.get("bbox")
        if bbox is None:
            return None
        return self.crop_and_resize_face(image, bbox)

    def preprocess_face(self, face: torch.Tensor) -> torch.Tensor:
        """
        Preprocess face for ArcFace model.

        Args:
            face: (C, 112, 112) tensor, values in [0, 1]

        Returns:
            Normalized face tensor
        """
        # ArcFace expects: (x - 127.5) / 127.5 = 2x - 1
        # Our input is in [0, 1], so: 2 * x - 1
        face = face * 2 - 1
        return face

    def extract_embedding(self, face: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding (differentiable).

        Args:
            face: (C, 112, 112) preprocessed face tensor

        Returns:
            (512,) embedding tensor
        """
        # Ensure face is float32 to match recognizer weights
        # This prevents "Input type (float) and bias type (c10::BFloat16)" warnings
        recognizer_device = next(self.recognizer.parameters()).device
        face = face.unsqueeze(0).to(
            device=recognizer_device, dtype=self._model_dtype
        )  # (1, C, 112, 112)
        embedding = self.recognizer(face)  # (1, 512)
        return embedding.squeeze(0)

    def forward(
        self,
        ref_images: torch.Tensor,
        gen_images: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor:
        """
        Compute differentiable face identity loss.

        The loss is differentiable with respect to gen_images, allowing
        gradients to flow back for training.

        Args:
            ref_images: Reference images, shape (B, C, H, W), values in [0, 1]
            gen_images: Generated images, shape (B, C, H, W), values in [0, 1]
            return_details: If True, also return per-sample losses and valid mask

        Returns:
            loss: Scalar loss value (1 - cosine_similarity, averaged over valid pairs)
        """
        batch_size = ref_images.shape[0]
        device = gen_images.device

        # Detect faces in both reference and generated images
        ref_faces = self.detect_faces(ref_images)
        gen_faces = self.detect_faces(gen_images)

        losses = []
        valid_indices = []

        for i in range(batch_size):
            ref_info = ref_faces[i]
            gen_info = gen_faces[i]

            if ref_info is None or gen_info is None:
                continue

            # Crop/align and preprocess faces
            ref_face = self.extract_face(ref_images[i], ref_info)
            gen_face = self.extract_face(gen_images[i], gen_info)
            if ref_face is None or gen_face is None:
                continue

            ref_face = self.preprocess_face(ref_face)
            gen_face = self.preprocess_face(gen_face)

            # Extract embeddings (differentiable for gen_face)
            with torch.no_grad():
                ref_emb = self.extract_embedding(ref_face)
            gen_emb = self.extract_embedding(gen_face)  # Gradients flow here!

            # Normalize embeddings
            ref_emb = F.normalize(ref_emb, dim=0)
            gen_emb = F.normalize(gen_emb, dim=0)

            # Cosine similarity loss
            cos_sim = torch.dot(ref_emb, gen_emb)
            loss = 1.0 - cos_sim

            losses.append(loss)
            valid_indices.append(i)

        if len(losses) == 0:
            # No valid face pairs found
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_details:
                return (
                    zero_loss,
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, dtype=torch.bool, device=device),
                )
            return zero_loss

        # Average losses
        losses = torch.stack(losses)
        avg_loss = losses.mean()

        if return_details:
            per_sample_losses = torch.zeros(batch_size, device=device)
            per_sample_losses[valid_indices] = losses.detach()
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            valid_mask[valid_indices] = True
            return avg_loss, per_sample_losses, valid_mask

        return avg_loss

    def extract_multiscale_features(self, face: torch.Tensor) -> dict:
        """
        Extract multi-scale features from face (differentiable).

        Args:
            face: (C, 112, 112) preprocessed face tensor

        Returns:
            Dict with features from each layer and final embedding
        """
        recognizer_device = next(self.recognizer.parameters()).device
        face = face.unsqueeze(0).to(
            device=recognizer_device, dtype=self._model_dtype
        )  # (1, C, 112, 112)
        features = self.recognizer.forward_multiscale(face)
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in features.items()}

    def forward_multiscale(
        self,
        ref_images: torch.Tensor,
        gen_images: torch.Tensor,
        layer_weights: dict = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-scale differentiable face identity loss.

        This loss combines:
        1. Identity embedding loss (final 512-d vector)
        2. Multi-scale feature losses from intermediate layers

        Recommended layer_weights:
        - layer1 (coarse structure): 0.1
        - layer2 (mid-level): 0.2
        - layer3 (fine details): 0.3
        - layer4 (semantic): 0.4
        - embedding (identity): 1.0

        Args:
            ref_images: Reference images, shape (B, C, H, W), values in [0, 1]
            gen_images: Generated images, shape (B, C, H, W), values in [0, 1]
            layer_weights: Dict of weights for each layer.
                          Default: {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3,
                                    'layer4': 0.4, 'embedding': 1.0}
            return_details: If True, also return per-layer losses

        Returns:
            total_loss: Weighted sum of all losses
            If return_details: (total_loss, loss_dict) where loss_dict has per-layer losses
        """
        if layer_weights is None:
            # Default weights: deeper layers get higher weights (more identity-related)
            # Total effective weight ~2.0 (0.1+0.2+0.3+0.4+1.0)
            layer_weights = {
                "layer1": 0.1,  # Coarse face structure
                "layer2": 0.2,  # Mid-level features
                "layer3": 0.3,  # Fine facial details (eyes, nose, mouth)
                "layer4": 0.4,  # High-level semantic features
                "embedding": 1.0,  # Final identity embedding (most important)
            }

        batch_size = ref_images.shape[0]
        device = gen_images.device

        # Detect faces
        ref_faces = self.detect_faces(ref_images)
        gen_faces = self.detect_faces(gen_images)

        # Accumulate losses per layer
        layer_losses = {k: [] for k in layer_weights.keys()}
        valid_indices = []

        for i in range(batch_size):
            ref_info = ref_faces[i]
            gen_info = gen_faces[i]

            if ref_info is None or gen_info is None:
                continue

            # Crop/align and preprocess faces
            ref_face = self.extract_face(ref_images[i], ref_info)
            gen_face = self.extract_face(gen_images[i], gen_info)
            if ref_face is None or gen_face is None:
                continue

            ref_face = self.preprocess_face(ref_face)
            gen_face = self.preprocess_face(gen_face)

            # Extract multi-scale features
            with torch.no_grad():
                ref_feats = self.extract_multiscale_features(ref_face)
            gen_feats = self.extract_multiscale_features(
                gen_face
            )  # Gradients flow here!

            # Compute loss for each layer
            for layer_name in layer_weights.keys():
                ref_feat = ref_feats[layer_name]
                gen_feat = gen_feats[layer_name]

                # Use cosine distance for all layers (not MSE)
                # MSE on flattened normalized vectors is numerically squashed by 1/N
                ref_norm = F.normalize(ref_feat.flatten(), dim=0)
                gen_norm = F.normalize(gen_feat.flatten(), dim=0)
                loss = 1.0 - torch.dot(ref_norm, gen_norm)

                layer_losses[layer_name].append(loss)

            valid_indices.append(i)

        if len(valid_indices) == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_details:
                valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                return (
                    zero_loss,
                    {k: torch.tensor(0.0, device=device) for k in layer_weights.keys()},
                    valid_mask,
                )
            return zero_loss

        # Compute weighted total loss
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        for layer_name, weight in layer_weights.items():
            if layer_losses[layer_name]:
                layer_loss = torch.stack(layer_losses[layer_name]).mean()
                loss_dict[layer_name] = layer_loss
                total_loss = total_loss + weight * layer_loss

        if return_details:
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            valid_mask[valid_indices] = True
            return total_loss, loss_dict, valid_mask

        return total_loss


# ============================================================================
# Original Non-Differentiable Face Loss (for evaluation only)
# ============================================================================


class ArcFaceLoss(nn.Module):
    """
    ArcFace-based identity loss for face generation.

    Uses InsightFace to extract face embeddings and computes
    cosine similarity loss between reference and generated images.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,  # GPU id, -1 for CPU
        det_size: tuple = (640, 640),
        root: str = None,  # Custom model directory, e.g., "/path/to/models"
    ):
        """
        Initialize ArcFace loss module.

        Args:
            model_name: InsightFace model name (e.g., "buffalo_l", "buffalo_s")
            ctx_id: GPU device id, -1 for CPU
            det_size: Detection size for face detector
            root: Custom root directory for model storage. If None, uses default
                  ~/.insightface/models/. You can manually download models to this
                  directory, e.g., download buffalo_l.zip from
                  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
                  and extract to {root}/buffalo_l/
        """
        super().__init__()

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is required for ArcFace loss. "
                "Install it with: pip install insightface onnxruntime-gpu"
            )

        # Initialize InsightFace model
        # If root is specified, use it; otherwise FaceAnalysis uses default ~/.insightface/models/
        fa_kwargs = {
            "name": model_name,
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }
        if root is not None:
            fa_kwargs["root"] = root
        self.app = FaceAnalysis(**fa_kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Embedding dimension (512 for ArcFace)
        self.embedding_dim = 512

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array for InsightFace.

        Args:
            tensor: Image tensor of shape (C, H, W) or (H, W, C), values in [0, 1]

        Returns:
            numpy array of shape (H, W, C), values in [0, 255], BGR format
        """
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            # (C, H, W) -> (H, W, C)
            tensor = tensor.permute(1, 2, 0)

        # Convert to numpy and scale to 0-255
        img_np = tensor.detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # RGB to BGR for InsightFace
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_np

    def extract_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding from image tensor.

        Args:
            image: Image tensor of shape (C, H, W), values in [0, 1]

        Returns:
            Face embedding tensor of shape (512,), or None if no face detected
        """
        img_np = self._tensor_to_numpy(image)

        # Detect and extract face
        faces = self.app.get(img_np)

        if len(faces) == 0:
            return None

        # Use the largest/most confident face
        face = max(faces, key=lambda x: x.det_score)
        embedding = torch.from_numpy(face.embedding).float()

        return embedding

    def extract_batch_embeddings(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract face embeddings from a batch of images.

        Args:
            images: Batch of images, shape (B, C, H, W)

        Returns:
            embeddings: Face embeddings, shape (N, 512) where N <= B
            valid_mask: Boolean mask indicating which images had faces detected
        """
        batch_size = images.shape[0]
        embeddings = []
        valid_indices = []

        for i in range(batch_size):
            emb = self.extract_embedding(images[i])
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if len(embeddings) == 0:
            return None, None

        embeddings = torch.stack(embeddings)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)
        valid_mask[valid_indices] = True

        return embeddings, valid_mask

    def forward(
        self,
        ref_images: torch.Tensor,
        gen_images: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor:
        """
        Compute face identity loss between reference and generated images.

        Args:
            ref_images: Reference images, shape (B, C, H, W)
            gen_images: Generated images, shape (B, C, H, W)
            return_details: If True, also return per-sample losses and valid mask

        Returns:
            loss: Scalar loss value (1 - cosine_similarity, averaged over valid pairs)

        If return_details is True, returns tuple of (loss, per_sample_losses, valid_mask)
        """
        batch_size = ref_images.shape[0]
        device = ref_images.device

        losses = []
        valid_indices = []

        for i in range(batch_size):
            ref_emb = self.extract_embedding(ref_images[i])
            gen_emb = self.extract_embedding(gen_images[i])

            if ref_emb is not None and gen_emb is not None:
                # Compute cosine similarity
                ref_emb = ref_emb.to(device)
                gen_emb = gen_emb.to(device)

                # Normalize embeddings
                ref_emb = F.normalize(ref_emb, dim=0)
                gen_emb = F.normalize(gen_emb, dim=0)

                # Cosine similarity (1.0 for identical, -1.0 for opposite)
                cos_sim = torch.dot(ref_emb, gen_emb)

                # Loss is 1 - cos_sim (0 for identical, 2 for opposite)
                loss = 1.0 - cos_sim
                losses.append(loss)
                valid_indices.append(i)

        if len(losses) == 0:
            # No valid face pairs found, return zero loss
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_details:
                return (
                    zero_loss,
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, dtype=torch.bool, device=device),
                )
            return zero_loss

        # Stack and average losses
        losses = torch.stack(losses)
        avg_loss = losses.mean()

        if return_details:
            per_sample_losses = torch.zeros(batch_size, device=device)
            per_sample_losses[valid_indices] = losses
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            valid_mask[valid_indices] = True
            return avg_loss, per_sample_losses, valid_mask

        return avg_loss


class FaceLossWrapper(nn.Module):
    """
    Wrapper for face loss that handles decoding latents to images.

    This wrapper takes care of:
    1. Decoding generated latents to pixel space
    2. Extracting the right portion of diptych images
    3. Computing face identity loss
    """

    def __init__(
        self,
        vae,
        arcface_model_name: str = "buffalo_l",
        image_size: int = 768,
        arcface_root: str = None,
    ):
        """
        Initialize face loss wrapper.

        Args:
            vae: VAE decoder for latent -> image conversion
            arcface_model_name: InsightFace model name
            image_size: Size of individual images (diptych is 2x this width)
            arcface_root: Custom root directory for InsightFace model storage.
                          If None, uses default ~/.insightface/models/
        """
        super().__init__()
        self.vae = vae
        self.arcface_loss = ArcFaceLoss(
            model_name=arcface_model_name, root=arcface_root
        )
        self.image_size = image_size

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using VAE.

        Args:
            latents: Latent tensor

        Returns:
            Decoded images, shape (B, C, H, W), values in [0, 1]
        """
        # VAE decoding
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        # Clamp to valid range
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)

        return images

    def extract_target_region(self, diptych_images: torch.Tensor) -> torch.Tensor:
        """
        Extract the target (right) region from diptych images.

        Args:
            diptych_images: Diptych images, shape (B, C, H, W*2)

        Returns:
            Target images, shape (B, C, H, W)
        """
        # Diptych structure: [ref_image | target_image]
        # Extract right half
        width = diptych_images.shape[-1] // 2
        return diptych_images[..., width:]

    def forward(
        self,
        ref_images: torch.Tensor,
        pred_latents: torch.Tensor,
        x_0_latents: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute face identity loss.

        Args:
            ref_images: Reference images (cropped faces), shape (B, C, H, W)
            pred_latents: Predicted latent (velocity or noise)
            x_0_latents: Original x_0 latents (ground truth)
            t: Timestep

        Returns:
            Face identity loss
        """
        # For now, we reconstruct x_0 estimate from prediction
        # This is a simplified version - the actual reconstruction depends on
        # the parameterization used (v-prediction, epsilon-prediction, etc.)

        # Decode ground truth latents to get target images
        with torch.no_grad():
            target_images = self.decode_latents(x_0_latents)
            target_images = self.extract_target_region(target_images)

        # Compute face loss between reference and target
        # Note: ref_images should already be the cropped reference faces
        loss = self.arcface_loss(ref_images, target_images)

        return loss


class DifferentiableFaceLossWrapper(nn.Module):
    """
    Differentiable wrapper for face loss that handles decoding latents to images.

    This wrapper is designed for LoRA training where gradients need to flow
    through the face loss back to the model. It uses DifferentiableFaceLoss
    with PyTorch ArcFace backbone.

    Key differences from FaceLossWrapper:
    - Uses DifferentiableFaceLoss instead of ArcFaceLoss
    - Gradients flow through the generated images
    - Requires pretrained ArcFace weights
    """

    def __init__(
        self,
        vae,
        arcface_weights_path: str = None,
        model_name: str = "r50",
        image_size: int = 768,
        device: str = "cuda",
        root: str = None,
        align_mode: str = "bbox",
    ):
        """
        Initialize differentiable face loss wrapper.

        Args:
            vae: VAE decoder for latent -> image conversion
            arcface_weights_path: Path to pretrained ArcFace weights (.pt file).
                                  Download from: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
            model_name: Model architecture ("r50" or "r100")
            image_size: Size of individual images (diptych is 2x this width)
            device: Device to run on ("cuda" or "cpu")
            root: Root directory for InsightFace detector model storage
            align_mode: Face alignment mode ("bbox" or "kps")
        """
        super().__init__()
        self.vae = vae
        self.face_loss = DifferentiableFaceLoss(
            model_name=model_name,
            pretrained_path=arcface_weights_path,
            device=device,
            root=root,
            align_mode=align_mode,
        )
        self.image_size = image_size

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using VAE.

        Note: This is NOT wrapped with torch.no_grad() to allow gradients
        to flow through for training.

        Args:
            latents: Latent tensor

        Returns:
            Decoded images, shape (B, C, H, W), values in [0, 1]
        """
        # VAE decoding
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        # Clamp to valid range
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)

        return images

    def extract_target_region(self, diptych_images: torch.Tensor) -> torch.Tensor:
        """
        Extract the target (right) region from diptych images.

        Args:
            diptych_images: Diptych images, shape (B, C, H, W*2)

        Returns:
            Target images, shape (B, C, H, W)
        """
        width = diptych_images.shape[-1] // 2
        return diptych_images[..., width:]

    def forward(
        self,
        ref_images: torch.Tensor,
        gen_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable face identity loss.

        Gradients flow through gen_latents -> VAE decode -> face recognition,
        allowing the model to optimize for face identity preservation.

        Args:
            ref_images: Reference images, shape (B, C, H, W), values in [0, 1]
            gen_latents: Generated latents from the model (gradients will flow through)

        Returns:
            Face identity loss (scalar)
        """
        # Decode latents (differentiable)
        gen_images = self.decode_latents(gen_latents)
        gen_images = self.extract_target_region(gen_images)

        # Compute face loss (differentiable through gen_images)
        loss = self.face_loss(ref_images, gen_images)

        return loss
