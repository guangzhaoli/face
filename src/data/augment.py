"""
Reference Image Augmentation Module

对参考图 (ref) 进行颜色、亮度、对比度、色温等增强，
打破 ref 和 target 之间的像素级对应关系，
迫使模型学习高级身份特征而非简单复制。

注意：不对 ref 图像进行模糊处理，以保持清晰的身份信息。
"""

import torch
import torch.nn.functional as F
import random
from typing import Optional, Dict, Any


class RefImageAugmentor:
    """
    参考图像增强器

    支持的增强类型：
    - 颜色抖动 (色相、饱和度)
    - 亮度调整
    - 对比度调整
    - 色温调整 (冷暖色调)
    - Gamma 校正

    所有增强都是可配置的，可以通过配置文件开关。
    """

    def __init__(
        self,
        # 总开关
        enabled: bool = True,
        # 各增强概率
        color_jitter_prob: float = 0.5,
        brightness_prob: float = 0.5,
        contrast_prob: float = 0.5,
        color_temp_prob: float = 0.3,
        gamma_prob: float = 0.3,
        saturation_prob: float = 0.4,
        # 增强强度
        hue_range: float = 0.05,  # 色相偏移范围 [-hue_range, +hue_range]
        saturation_range: tuple = (0.7, 1.3),  # 饱和度范围
        brightness_range: tuple = (0.7, 1.3),  # 亮度范围
        contrast_range: tuple = (0.7, 1.3),  # 对比度范围
        color_temp_range: tuple = (-0.15, 0.15),  # 色温偏移
        gamma_range: tuple = (0.8, 1.2),  # Gamma 范围
    ):
        """
        初始化增强器

        Args:
            enabled: 总开关，False 时不进行任何增强
            color_jitter_prob: 颜色抖动（色相）的应用概率
            brightness_prob: 亮度调整的应用概率
            contrast_prob: 对比度调整的应用概率
            color_temp_prob: 色温调整的应用概率
            gamma_prob: Gamma 校正的应用概率
            saturation_prob: 饱和度调整的应用概率
            hue_range: 色相偏移范围
            saturation_range: 饱和度调整范围
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围
            color_temp_range: 色温调整范围
            gamma_range: Gamma 校正范围
        """
        self.enabled = enabled
        self.color_jitter_prob = color_jitter_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.color_temp_prob = color_temp_prob
        self.gamma_prob = gamma_prob
        self.saturation_prob = saturation_prob

        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.color_temp_range = color_temp_range
        self.gamma_range = gamma_range

    @staticmethod
    def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
        """
        将 RGB 图像转换为 HSV 颜色空间

        Args:
            image: RGB 图像 (C, H, W) 或 (B, C, H, W), 值域 [0, 1]

        Returns:
            HSV 图像，同样形状
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        r, g, b = image[:, 0], image[:, 1], image[:, 2]

        max_val, max_idx = image.max(dim=1)
        min_val, _ = image.min(dim=1)
        diff = max_val - min_val

        # Hue
        h = torch.zeros_like(max_val)
        mask = diff > 0

        # Red is max
        r_mask = mask & (max_idx == 0)
        h[r_mask] = (60 * (g[r_mask] - b[r_mask]) / diff[r_mask]) % 360

        # Green is max
        g_mask = mask & (max_idx == 1)
        h[g_mask] = 60 * (b[g_mask] - r[g_mask]) / diff[g_mask] + 120

        # Blue is max
        b_mask = mask & (max_idx == 2)
        h[b_mask] = 60 * (r[b_mask] - g[b_mask]) / diff[b_mask] + 240

        h = h / 360  # Normalize to [0, 1]

        # Saturation
        s = torch.zeros_like(max_val)
        s[max_val > 0] = diff[max_val > 0] / max_val[max_val > 0]

        # Value
        v = max_val

        hsv = torch.stack([h, s, v], dim=1)

        if squeeze:
            hsv = hsv.squeeze(0)

        return hsv

    @staticmethod
    def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        """
        将 HSV 图像转换回 RGB 颜色空间

        Args:
            hsv: HSV 图像 (C, H, W) 或 (B, C, H, W)

        Returns:
            RGB 图像，同样形状，值域 [0, 1]
        """
        if hsv.dim() == 3:
            hsv = hsv.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        h, s, v = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]

        c = v * s
        x = c * (1 - torch.abs((h / 60) % 2 - 1))
        m = v - c

        h_sector = (h / 60).long() % 6

        rgb = torch.zeros_like(hsv)

        # Sector 0
        mask = h_sector == 0
        rgb[:, 0][mask] = c[mask]
        rgb[:, 1][mask] = x[mask]

        # Sector 1
        mask = h_sector == 1
        rgb[:, 0][mask] = x[mask]
        rgb[:, 1][mask] = c[mask]

        # Sector 2
        mask = h_sector == 2
        rgb[:, 1][mask] = c[mask]
        rgb[:, 2][mask] = x[mask]

        # Sector 3
        mask = h_sector == 3
        rgb[:, 1][mask] = x[mask]
        rgb[:, 2][mask] = c[mask]

        # Sector 4
        mask = h_sector == 4
        rgb[:, 0][mask] = x[mask]
        rgb[:, 2][mask] = c[mask]

        # Sector 5
        mask = h_sector == 5
        rgb[:, 0][mask] = c[mask]
        rgb[:, 2][mask] = x[mask]

        rgb = rgb + m.unsqueeze(1)

        if squeeze:
            rgb = rgb.squeeze(0)

        return rgb.clamp(0, 1)

    def adjust_hue(self, image: torch.Tensor) -> torch.Tensor:
        """
        调整色相

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            色相调整后的图像
        """
        hue_shift = random.uniform(-self.hue_range, self.hue_range)

        hsv = self.rgb_to_hsv(image)
        hsv[0] = (hsv[0] + hue_shift) % 1.0
        return self.hsv_to_rgb(hsv)

    def adjust_saturation(self, image: torch.Tensor) -> torch.Tensor:
        """
        调整饱和度

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            饱和度调整后的图像
        """
        factor = random.uniform(*self.saturation_range)

        hsv = self.rgb_to_hsv(image)
        hsv[1] = (hsv[1] * factor).clamp(0, 1)
        return self.hsv_to_rgb(hsv)

    def adjust_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """
        调整亮度

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            亮度调整后的图像
        """
        factor = random.uniform(*self.brightness_range)
        return (image * factor).clamp(0, 1)

    def adjust_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """
        调整对比度

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            对比度调整后的图像
        """
        factor = random.uniform(*self.contrast_range)

        # 计算灰度均值
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        mean_gray = gray.mean()

        # 对比度调整
        adjusted = (image - mean_gray) * factor + mean_gray
        return adjusted.clamp(0, 1)

    def adjust_color_temperature(self, image: torch.Tensor) -> torch.Tensor:
        """
        调整色温 (冷暖色调)

        正值偏暖 (增加红色/黄色)，负值偏冷 (增加蓝色)

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            色温调整后的图像
        """
        temp_shift = random.uniform(*self.color_temp_range)

        result = image.clone()
        # 暖色：增加红色，减少蓝色
        # 冷色：减少红色，增加蓝色
        result[0] = (result[0] + temp_shift).clamp(0, 1)  # R
        result[2] = (result[2] - temp_shift).clamp(0, 1)  # B

        return result

    def adjust_gamma(self, image: torch.Tensor) -> torch.Tensor:
        """
        Gamma 校正

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            Gamma 校正后的图像
        """
        gamma = random.uniform(*self.gamma_range)
        return image.pow(gamma).clamp(0, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对输入图像应用随机增强

        Args:
            image: RGB 图像 (C, H, W), 值域 [0, 1]

        Returns:
            增强后的图像
        """
        if not self.enabled:
            return image

        # 创建副本避免修改原图
        augmented = image.clone()

        # 随机应用各种增强
        if random.random() < self.color_jitter_prob:
            augmented = self.adjust_hue(augmented)

        if random.random() < self.saturation_prob:
            augmented = self.adjust_saturation(augmented)

        if random.random() < self.brightness_prob:
            augmented = self.adjust_brightness(augmented)

        if random.random() < self.contrast_prob:
            augmented = self.adjust_contrast(augmented)

        if random.random() < self.color_temp_prob:
            augmented = self.adjust_color_temperature(augmented)

        if random.random() < self.gamma_prob:
            augmented = self.adjust_gamma(augmented)

        return augmented

    @classmethod
    def from_config(
        cls, config: Optional[Dict[str, Any]] = None
    ) -> "RefImageAugmentor":
        """
        从配置字典创建增强器

        Args:
            config: 增强配置字典，如果为 None 则返回默认增强器

        Returns:
            RefImageAugmentor 实例

        配置示例:
            ref_augment:
              enabled: true
              color_jitter_prob: 0.5
              brightness_prob: 0.5
              contrast_prob: 0.5
              color_temp_prob: 0.3
              gamma_prob: 0.3
              saturation_prob: 0.4
              hue_range: 0.05
              saturation_range: [0.7, 1.3]
              brightness_range: [0.7, 1.3]
              contrast_range: [0.7, 1.3]
              color_temp_range: [-0.15, 0.15]
              gamma_range: [0.8, 1.2]
        """
        if config is None:
            return cls(enabled=False)

        return cls(
            enabled=config.get("enabled", True),
            color_jitter_prob=config.get("color_jitter_prob", 0.5),
            brightness_prob=config.get("brightness_prob", 0.5),
            contrast_prob=config.get("contrast_prob", 0.5),
            color_temp_prob=config.get("color_temp_prob", 0.3),
            gamma_prob=config.get("gamma_prob", 0.3),
            saturation_prob=config.get("saturation_prob", 0.4),
            hue_range=config.get("hue_range", 0.05),
            saturation_range=tuple(config.get("saturation_range", [0.7, 1.3])),
            brightness_range=tuple(config.get("brightness_range", [0.7, 1.3])),
            contrast_range=tuple(config.get("contrast_range", [0.7, 1.3])),
            color_temp_range=tuple(config.get("color_temp_range", [-0.15, 0.15])),
            gamma_range=tuple(config.get("gamma_range", [0.8, 1.2])),
        )

    def __repr__(self) -> str:
        return (
            f"RefImageAugmentor(\n"
            f"  enabled={self.enabled},\n"
            f"  color_jitter_prob={self.color_jitter_prob},\n"
            f"  saturation_prob={self.saturation_prob},\n"
            f"  brightness_prob={self.brightness_prob},\n"
            f"  contrast_prob={self.contrast_prob},\n"
            f"  color_temp_prob={self.color_temp_prob},\n"
            f"  gamma_prob={self.gamma_prob}\n"
            f")"
        )
