# Face Identity Loss (ArcFace) for LoRA Training

本文档介绍如何使用 ArcFace 人脸身份损失来提升人脸生成质量。

## 概述

本模块提供两种 Face Loss 实现：

| 类型 | 类名 | 梯度 | 用途 |
|------|------|------|------|
| **可微分** | `DifferentiableFaceLoss` | ✅ 支持 | LoRA 训练优化 |
| **不可微分** | `ArcFaceLoss` | ❌ 不支持 | 评估/推理监控 |

> ⚠️ **重要**: 如果要用 Face Loss 来**优化训练**（使生成的人脸更像参考图），必须使用 `DifferentiableFaceLoss`。原始的 `ArcFaceLoss` 使用 ONNX 推理，梯度无法回传！

---

## 架构设计

### 梯度流程对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    DifferentiableFaceLoss                       │
│                                                                 │
│  gen_latents ──► VAE Decode ──► Face Crop ──► IResNet ──► Loss  │
│       │              │             │            │          │    │
│       └──────────────┴─────────────┴────────────┴──────────┘    │
│                    ↑ 梯度可以一直回传 ↑                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       ArcFaceLoss (原始)                         │
│                                                                 │
│  gen_images ──► numpy ──► InsightFace ONNX ──► Loss             │
│       │           ✗            ✗               │                │
│       │       梯度断开      梯度断开            │                │
└─────────────────────────────────────────────────────────────────┘
```

### 模块组成

```
src/models/arcface_loss.py
├── IResNet / IBasicBlock          # PyTorch ArcFace 骨干网络
├── iresnet50() / iresnet100()     # 模型构造函数
│
├── DifferentiableFaceLoss         # 可微分人脸 Loss (训练用)
├── DifferentiableFaceLossWrapper  # 封装 VAE 解码 + Loss
│
├── ArcFaceLoss                    # 不可微分 Loss (评估用)
└── FaceLossWrapper                # 封装 VAE 解码 + Loss
```

---

## 安装依赖

```bash
# 基础依赖 (人脸检测用)
pip install insightface onnxruntime-gpu

# 可选: 如果遇到 onnxruntime 问题
pip install onnxruntime-gpu==1.16.0
```

### 下载预训练权重

**DifferentiableFaceLoss 需要 PyTorch ArcFace 权重：**

从 InsightFace 官方下载：
- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

推荐权重：
- `glint360k_r50.pth` (IResNet-50, 推荐)
- `glint360k_r100.pth` (IResNet-100, 更精确但更慢)

**InsightFace 检测模型（自动下载）：**

首次运行时会自动下载 `buffalo_l` 模型（约400MB）到 `~/.insightface/models/`

手动下载：
```bash
# 下载地址: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
# 解压到: ~/.insightface/models/buffalo_l/
```

---

## 使用方法

### 方法 1: 使用 DifferentiableFaceLossWrapper（推荐）

最简单的方式，封装了 VAE 解码和人脸 Loss：

```python
from src.models.arcface_loss import DifferentiableFaceLossWrapper

# 初始化
face_loss_wrapper = DifferentiableFaceLossWrapper(
    vae=vae,                                    # VAE 解码器
    arcface_weights_path="path/to/backbone.pth", # PyTorch 权重路径
    model_name="r50",                           # "r50" 或 "r100"
    device="cuda",
    root="/path/to/insightface/models",         # 可选: 自定义检测模型目录
)

# 在训练循环中使用
def training_step(batch):
    ref_images = batch["ref_images"]      # 参考人脸图像 [B, C, H, W]
    gen_latents = model(...)              # 模型生成的 latents
    
    # 计算可微分的人脸 Loss
    face_loss = face_loss_wrapper(ref_images, gen_latents)
    
    # 总损失
    total_loss = mse_loss + 0.1 * face_loss
    total_loss.backward()  # ✅ 梯度正常回传
```

### 方法 2: 直接使用 DifferentiableFaceLoss

如果已经有解码后的图像：

```python
from src.models.arcface_loss import DifferentiableFaceLoss

face_loss = DifferentiableFaceLoss(
    model_name="r50",
    pretrained_path="path/to/backbone.pth",
    device="cuda",
)

# 输入: [0, 1] 范围的图像
loss = face_loss(
    ref_images=ref_images,    # 参考图像 [B, C, H, W]
    gen_images=gen_images,    # 生成图像 [B, C, H, W] (梯度会回传到这里)
)
```

### 方法 3: 评估/监控用 ArcFaceLoss

只用于计算指标，不参与训练：

```python
from src.models.arcface_loss import ArcFaceLoss

arcface = ArcFaceLoss(
    model_name="buffalo_l",
    root="/custom/path",  # 可选
)

with torch.no_grad():
    loss = arcface(ref_images, gen_images)
    print(f"Face similarity: {1 - loss.item():.3f}")
```

---

## 训练配置示例

```yaml
model:
  use_face_loss: true              # 启用人脸损失
  face_loss_weight: 0.1            # 损失权重
  face_align_mode: "bbox"          # "bbox" 或 "kps" (关键点对齐)
  face_loss_type: "differentiable" # 使用可微分版本
  arcface_weights: "path/to/backbone.pth"
  arcface_model: "r50"             # r50 或 r100
```

### 数据类型要求

Face Loss **仅对 `data_type == "person_head"` 的数据生效**：

| data_type | 是否计算 Face Loss |
|-----------|-------------------|
| `person` | ✅ 计算 |
| `person_head` | ✅ 计算 |
| `person_hair` | ❌ 不计算 |
| 其他 | ❌ 不计算 |

---

## 对齐方式（可选）

- `face_align_mode: "bbox"`：基于人脸框裁剪 + resize（当前默认行为）
- `face_align_mode: "kps"`：基于 5 点关键点做相似变换对齐，再送入 ArcFace

关键点对齐对姿态/旋转更稳，通常能提升身份一致性；若关键点失败会自动回退到 bbox。

## 技术原理

### 训练流程（Flow Matching）

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Step                            │
└─────────────────────────────────────────────────────────────────┘

1. 采样时间步 t ~ sigmoid(N(0,1))  ∈ (0, 1)
2. 采样噪声 x_1 ~ N(0, I)
3. 插值得到 x_t = (1-t) * x_0 + t * x_1   (Flow Matching)
4. Transformer 预测 velocity: pred ≈ v = x_1 - x_0
5. MSE Loss: L_mse = ||pred - (x_1 - x_0)||²

┌─────────────────────────────────────────────────────────────────┐
│            Differentiable Face Loss (仅 person_head)            │
└─────────────────────────────────────────────────────────────────┘

6. 从 pred 反推 x_0 估计值:
   
   x_0_hat = x_t - t * pred
   
   (因为 x_t = x_0 + t*v, 所以 x_0 = x_t - t*v)

7. Unpack latents -> VAE Decode -> 预测图像
8. Face Loss: L_face = 1 - cos_sim(ref_embed, pred_embed)
9. Total Loss: L = L_mse + λ * L_face

梯度流向: L_face -> pred_images -> VAE -> x_0_hat -> pred -> Transformer -> LoRA
```

### 损失计算流程图

```
┌─────────────┐
│  ref_image  │ (参考人脸, 真值)
│   [B,C,H,W] │
└──────┬──────┘
       │
       │ InsightFace Detector (no_grad)
       ▼
   ref_bbox ──────────────────────────────────┐
       │                                      │
       │ Differentiable Crop                  │
       ▼                                      │
   ref_face                                   │
       │                                      │
       │ IResNet (frozen)                     │
       ▼                                      │
   ref_embed ◄────────────────────────────────┤
   (512-dim)                                  │
                                              │
┌─────────────┐                               │
│    pred     │ (Transformer 输出)             │
│  velocity   │                               │
└──────┬──────┘                               │
       │                                      │
       │ x_0_hat = x_t - t * pred             │
       ▼                                      │
   x_0_hat                                    │
       │                                      │
       │ Unpack Latents                       │
       ▼                                      │
   unpacked_latents                           │
       │                                      │
       │ VAE Decode (differentiable)          │
       ▼                                      │
   pred_images ──────────────────────────────►│
       │                                      │
       │ InsightFace Detector (no_grad)       │
       ▼                                      │
   gen_bbox ──────────────────────────────────┤
       │                                      │
       │ Differentiable Crop                  │
       ▼                                      │
   gen_face                                   │
       │                                      │
       │ IResNet (frozen)                     │
       ▼                                      │
   gen_embed ◄────────────────────────────────┘
   (512-dim)
       │
       └───────┬───────────┐
               ▼           ▼
      Face Loss = 1 - cos_similarity(ref_embed, gen_embed)
               │
               │ backward()
               ▼
      梯度流回 pred -> Transformer -> LoRA 参数
```

### 总损失公式

```
total_loss = mse_loss + face_loss_weight × face_loss
```

其中：
- `mse_loss`: 原始的扩散模型损失（预测 velocity）
- `face_loss`: 人脸身份损失（范围 0~2，越小越好）
- `face_loss_weight`: 权重系数（推荐 0.05 ~ 0.2）

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/models/arcface_loss.py` | ArcFace 损失模块实现 |
| `src/models/my_model.py` | 模型训练逻辑（集成 Face Loss） |
| `src/data/base.py` | 数据加载（传递 data_type） |

---

## 常见问题

### Q: 训练报错 bus error / shm 不足？

A: Docker 容器共享内存不足，添加参数：
```bash
docker run --shm-size=8g ...
```
或减少 DataLoader workers：
```python
num_workers=0
```

### Q: 启用后训练变慢了？

A: 是的，Face Loss 需要：
1. 从 velocity 预测 x_0
2. VAE 解码（计算密集）
3. 人脸检测
4. IResNet 前向传播

预计增加 30-50% 训练时间。

### Q: 检测不到人脸怎么办？

A: 系统会自动跳过该样本的 Face Loss，返回 0 损失并打印警告。

### Q: 权重如何设置？

A: 建议从 0.1 开始：
- **0.05**: 轻微人脸约束
- **0.1**: 推荐起始值
- **0.2**: 强人脸约束（可能过拟合）

### Q: r50 和 r100 选哪个？

A: 
- `r50` (IResNet-50): 推荐，速度快，精度够用
- `r100` (IResNet-100): 更精确，但更慢、更占显存

### Q: 为什么需要两个模型（检测 + 识别）？

A:
- **检测模型** (InsightFace buffalo_l): 找到人脸位置，返回 bounding box
- **识别模型** (PyTorch IResNet): 提取人脸 embedding，需要可微分

