"""
AllDataset - 数据集加载模块

支持加载 Insert Anything 训练数据，包含：
- ref_image: 参考图像
- ref_mask: 参考区域 mask
- tar_image: 目标图像
- tar_mask: 目标区域 mask

支持对 ref_image 进行数据增强以减少 copy-paste 现象。
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .augment import RefImageAugmentor


class AllDataset(Dataset):
    """
    Insert Anything 训练数据集

    目录结构:
        image_dir/
        ├── ref_image/    # 参考图像
        ├── ref_mask/     # 参考区域 mask
        ├── tar_image/    # 目标图像
        └── tar_mask/     # 目标区域 mask
    """

    def __init__(
        self,
        image_dir: str,
        data_type: str = "unknown",
        image_size: int = 768,
        ref_augment_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化数据集

        Args:
            image_dir: 数据集根目录
            data_type: 数据类型标识 (person, person_head, person_hair, object, accessory)
            image_size: 图像大小
            ref_augment_config: 参考图像增强配置，如果为 None 则不增强
        """
        self.image_dir = Path(image_dir)
        self.data_type = data_type
        self.image_size = image_size

        # 初始化增强器
        self.ref_augmentor = RefImageAugmentor.from_config(ref_augment_config)

        # 获取所有样本
        self.samples = self._load_samples()

        print(
            f"[AllDataset] Loaded {len(self.samples)} samples from {image_dir} "
            f"(type={data_type}, augment={self.ref_augmentor.enabled})"
        )

    def _load_samples(self) -> List[str]:
        """
        加载所有有效样本的文件名

        Returns:
            文件名列表 (不含扩展名)
        """
        ref_mask_dir = self.image_dir / "ref_mask"
        if not ref_mask_dir.exists():
            print(f"[Warning] ref_mask directory not found: {ref_mask_dir}")
            return []

        samples = []
        for f in ref_mask_dir.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                filename = f.name
                # 检查其他三个文件是否存在
                ref_image = self.image_dir / "ref_image" / filename
                tar_image = self.image_dir / "tar_image" / filename
                tar_mask = self.image_dir / "tar_mask" / filename

                if ref_image.exists() and tar_image.exists() and tar_mask.exists():
                    samples.append(filename)

        return sorted(samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        加载图像并转换为 tensor

        Args:
            path: 图像路径

        Returns:
            图像 tensor (C, H, W), 值域 [0, 1]
        """
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img_tensor

    def _load_mask(self, path: Path) -> torch.Tensor:
        """
        加载 mask 并转换为 tensor

        Args:
            path: mask 路径

        Returns:
            mask tensor (1, H, W), 值域 [0, 1]
        """
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (H, W) -> (1, H, W)
        return mask_tensor

    def _create_diptych(
        self,
        ref_image: torch.Tensor,
        tar_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        创建 diptych 图像 (左边参考图，右边目标图)

        Args:
            ref_image: 参考图像 (C, H, W)
            tar_image: 目标图像 (C, H, W)

        Returns:
            diptych 图像 (C, H, W*2)
        """
        return torch.cat([ref_image, tar_image], dim=2)

    def _create_diptych_mask(
        self,
        tar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        创建 diptych mask (左边全黑，右边目标 mask)

        Args:
            tar_mask: 目标 mask (1, H, W)

        Returns:
            diptych mask (1, H, W*2)
        """
        zeros = torch.zeros_like(tar_mask)
        return torch.cat([zeros, tar_mask], dim=2)

    def _create_src_image(
        self,
        ref_image: torch.Tensor,
        tar_image: torch.Tensor,
        tar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        创建 source 图像 (用于 inpainting 条件)

        左边是参考图，右边是被 mask 遮挡的目标图

        Args:
            ref_image: 参考图像 (C, H, W)
            tar_image: 目标图像 (C, H, W)
            tar_mask: 目标 mask (1, H, W)

        Returns:
            source 图像 (C, H, W*2)
        """
        # 目标区域被遮挡 (用 0.5 灰色或其他值)
        masked_tar = tar_image * (1 - tar_mask)
        return torch.cat([ref_image, masked_tar], dim=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个样本

        Returns:
            dict 包含:
                - result: diptych 图像 (C, H, W*2)
                - src: source 图像 (C, H, W*2)
                - mask: diptych mask (1, H, W*2)
                - ref: 参考图像 (C, H, W)，用于 face loss
                - data_type: 数据类型
        """
        filename = self.samples[idx]

        # 加载图像和 mask
        ref_image = self._load_image(self.image_dir / "ref_image" / filename)
        tar_image = self._load_image(self.image_dir / "tar_image" / filename)
        tar_mask = self._load_mask(self.image_dir / "tar_mask" / filename)

        # 对参考图进行增强 (用于训练，打破像素级对应)
        ref_image_augmented = self.ref_augmentor(ref_image)

        # 创建 diptych
        result = self._create_diptych(ref_image, tar_image)
        src = self._create_src_image(ref_image_augmented, tar_image, tar_mask)
        mask = self._create_diptych_mask(tar_mask)

        return {
            "result": result,  # GT diptych
            "src": src,  # 输入 (带 mask 的 diptych)
            "mask": mask,  # diptych mask
            "ref": ref_image,  # 原始参考图 (用于 face loss)
            "data_type": self.data_type,
        }

    def __add__(self, other: "AllDataset") -> "ConcatDataset":
        """
        支持数据集拼接: dataset1 + dataset2

        Args:
            other: 另一个数据集

        Returns:
            拼接后的数据集
        """
        return ConcatDataset([self, other])


class ConcatDataset(Dataset):
    """
    拼接多个数据集
    """

    def __init__(self, datasets: List[AllDataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        cumsum = 0
        for d in datasets:
            cumsum += len(d)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                dataset_idx = i
                break

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    def __add__(self, other: "AllDataset") -> "ConcatDataset":
        if isinstance(other, ConcatDataset):
            return ConcatDataset(self.datasets + other.datasets)
        return ConcatDataset(self.datasets + [other])
