# MLX 版本的 SAM 3.1 (Segment Anything Model 3.1)

[Meta's SAM 3.1](https://github.com/facebookresearch/sam3) 的 MLX 移植版本——扩展 SAM 3，具有 **Object Multiplex** 功能，用于更快的多对象跟踪（128 对象时约 7 倍）。

> **注意：** SAM 3.1 共享与 SAM 3 相同的检测管道，但添加了三头 FPN、多路复用掩码解码器（同时处理 16 个对象）和用于跟踪的解耦内存注意力。

## SAM 3.1 的新功能

| 组件 | SAM 3 | SAM 3.1 |
|-----------|-------|---------|
| FPN 颈部 | 2 个头, 4 个尺度 | **3 个头** (检测、交互、传播), **3 个尺度** |
| 跟踪器掩码解码器 | 一次处理 1 个对象 | **同时处理 16 个对象** (MultiplexMaskDecoder) |
| 内存注意力 | 标准变换器 | **解耦**，带图像交叉注意力 + RoPE |
| 跟踪器 | 单个解码器 | **双解码器** (交互 + 传播) |
| 检测分数 | 0.87, 0.82 (猫) | **0.90, 0.86** (猫) — 重新训练的权重 |

## 快速开始

```python
from PIL import Image
from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor

model_path = get_model_path("mlx-community/sam3.1-bf16")
model = load_model(model_path)
processor = Sam31Processor.from_pretrained(str(model_path))
predictor = Sam3Predictor(model, processor, score_threshold=0.3)
```

## 目标检测

```python
image = Image.open("photo.jpg")
result = predictor.predict(image, text_prompt="a dog")

for i in range(len(result.scores)):
    x1, y1, x2, y2 = result.boxes[i]
    print(f"[{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

## 实例分割

```python
result = predictor.predict(image, text_prompt="a person")

# result.boxes   -> (N, 4) xyxy 边界框
# result.masks   -> (N, H, W) 二进制分割掩码
# result.scores  -> (N,) 置信度分数

import numpy as np
overlay = np.array(image).copy()
W, H = image.size
for i in range(len(result.scores)):
    mask = result.masks[i]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)))
    binary = mask > 0
    overlay[binary] = (overlay[binary] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
```

## 框引导检测

```python
import numpy as np
boxes = np.array([[100, 50, 400, 350]])
result = predictor.predict(image, text_prompt="a cat", boxes=boxes)
```

## 注释器

15 个内置注释器用于可视化。可用 `+` 链接，无需外部依赖。

```python
from mlx_vlm.models.sam3.annotators import (
    MaskAnnotator, BoxAnnotator, LabelAnnotator,
    BoxCornerAnnotator, RoundBoxAnnotator, EllipseAnnotator,
    HaloAnnotator, ColorAnnotator, BackgroundOverlayAnnotator,
    BlurAnnotator, PixelateAnnotator, PercentageBarAnnotator,
    TriangleAnnotator, DotAnnotator, CircleAnnotator,
)

frame = np.array(image)[..., ::-1]  # RGB->BGR

# 链接注释器
annotator = MaskAnnotator(opacity=0.4) + BoxAnnotator() + LabelAnnotator()
out = annotator.annotate(frame, result)

# 或混合搭配
annotator = HaloAnnotator() + BoxCornerAnnotator() + PercentageBarAnnotator()
out = annotator.annotate(frame, result)

# 隐私模式
out = BlurAnnotator(kernel_size=31).annotate(frame, result)
```

| 快速 (<1ms) | 中等 (1-3ms) | 基于掩码 (~10ms) |
|------------|----------------|-------------------|
| Box, BoxCorner, RoundBox | Blur, Pixelate, Color | Mask, Halo, BgOverlay |
| Ellipse, Circle, Dot | | |
| Triangle, Label, PercentBar | | |

## CLI

SAM 3.1 有自己的 CLI，带有优化的实时模式：

```bash
# 目标检测
python -m mlx_vlm.models.sam3_1.generate --task detect --image photo.jpg --prompt "a cat" --model mlx-community/sam3.1-bf16

# 实例分割
python -m mlx_vlm.models.sam3_1.generate --image photo.jpg --prompt "a cat" --model mlx-community/sam3.1-bf16

# 视频跟踪
python -m mlx_vlm.models.sam3_1.generate --task track --video input.mp4 --prompt "a car" --model mlx-community/sam3.1-bf16

# 实时网络摄像头（优化：骨干缓存 + 跟踪器传播）
python -m mlx_vlm.models.sam3_1.generate --task realtime --prompt "a person" --model mlx-community/sam3.1-bf16 --resolution 224
```

### 优化的实时模式

SAM 3.1 实时管道使用两种优化来加快推理：

1. **骨干缓存**：ViT 骨干（224px 时约 67ms，1008px 时约 783ms）在中间帧中重用，仅在缓存的特征上运行轻量级 DETR 头。
2. **跟踪器传播**（仅 1008px）：在完整检测之间，多路复用跟踪器使用内存注意力 + 掩码解码器传播掩码，而不是重新运行 DETR。

```bash
# 调整优化参数
python -m mlx_vlm.models.sam3_1.generate --task realtime --prompt "a person" \
  --model mlx-community/sam3.1-bf16 --resolution 224 \
  --backbone-every 5 \    # 每 N 帧重新运行 ViT（默认：5）
  --detect-every 15 \     # 每 N 帧重新运行完整 DETR（默认：15）
  --memory-every 3        # 每 N 个传播帧更新跟踪器内存（默认：3）
```

## 架构

```
图像 (1008x1008)
  |
  v
ViT 骨干 (32 层, 1024d, 窗口 + 全局注意力, 2D RoPE)
  |
  v
TriViTDetNeck (3 个并行 FPN 头, 3 个尺度: 288x288, 144x144, 72x72)
  |                    |                      |
  v                    v                      v
检测 FPN        交互 FPN              传播 FPN
  |                    |                      |
  v                    v                      v
DETR 编码器      交互 SAM               多路复用跟踪器
  + 解码器        掩码解码器              同时处理 16 个对象
  200 查询         (点击/框)
  |
  +---> DotProductScoring --> pred_logits
  +---> BoxHead           --> pred_boxes (xyxy)
  +---> MaskDecoder/FPN   --> pred_masks (288x288)
```

### 组件

| 组件 | 描述 | 权重前缀 |
|-----------|-------------|---------------|
| 视觉编码器 | ViT-L + **TriViTDetNeck** (3 个头) | `detector_model.vision_encoder.*` |
| 文本编码器 | CLIP (24L, 1024d) | `detector_model.text_encoder.*` |
| DETR 编码器 | 6 层 pre-norm，带文本交叉注意力 | `detector_model.detr_encoder.*` |
| DETR 解码器 | 6 层 post-norm，200 查询，BoxRPB | `detector_model.detr_decoder.*` |
| 几何编码器 | 框/点提示词编码 | `detector_model.geometry_encoder.*` |
| 掩码解码器 | 像素解码器 + 实例投影 | `detector_model.mask_decoder.*` |
| 点积评分 | 文本-查询分类器 | `detector_model.dot_product_scoring.*` |
| **多路复用掩码解码器** | **同时处理 16 个对象** | `tracker_model.sam_mask_decoder.*` |
| **交互式 SAM** | **点击/框提示词解码器** | `tracker_model.interactive_sam_*` |
| **解耦内存注意力** | **4 层，带图像交叉注意力** | `tracker_model.memory_attention.*` |
| 内存编码器 | 多路复用感知（32 通道掩码输入） | `tracker_model.memory_encoder.*` |

## 文件结构

```
mlx_vlm/models/sam3_1/
├── __init__.py           # 模块导出
├── config.py             # 配置（扩展 SAM 3，带 multiplex_count、3 个尺度）
├── vision.py             # TriViTDetNeck（从 sam3 导入 ViT 骨干）
├── sam_components.py     # MultiplexMaskDecoder、DecoupledMemoryAttention、SimpleRoPEAttention
├── tracker.py            # MultiplexTrackerModel（双解码器、多路复用嵌入）
├── sam3_1.py             # 主模型 + 清理
├── processing_sam3_1.py  # 处理器（与 SAM 3 相同）
├── generate.py           # 推理管道（优化的实时模式，带骨干缓存 + 跟踪器）
└── convert_weights.py    # Meta .pt → MLX safetensors 转换器
```

### 从 SAM 3 复用的代码

SAM 3.1 直接从 `sam3/` 导入这些模块（无重复）：
- ViT 骨干、patch 嵌入、窗口注意力、2D RoPE (`sam3.vision`)
- CLIP 文本编码器 (`sam3.text_encoder`)
- DETR 编码器 + 解码器 (`sam3.encoder`, `sam3.decoder`)
- 位置编码 (`sam3.position`)
- 几何编码器 (`sam3.geometry`)
- 检测器分割头 (`sam3.segmentation`)
- SAM 提示词编码器、TwoWayTransformer (`sam3.sam_components`)

## 基准测试 (Apple Silicon)

在 M3 Max 上测量，使用 MLX、bf16 精度。SAM 3.1 的多路复用跟踪器每帧为所有对象调用 `track_step` **一次**，而 SAM 3 每个对象调用它 **一次**。

### 检测速度

检测使用相同的 DETR 管道——大致相同的速度：

| 任务 | SAM 3 | SAM 3.1 | 说明 |
|------|-------|---------|-------|
| 单个提示词 | 998ms | 1036ms | ~相同（ViT 骨干占主导地位） |
| 2 个提示词 | 1202ms | 1234ms | ~相同 |
| 5 个提示词 | 1746ms | 1796ms | ~相同 |

### 检测精度

SAM 3.1 有改进的权重——更高的分数和更少的误报：

| 提示词 | SAM 3 | SAM 3.1 |
|--------|-------|---------|
| "a cat" (2 只猫) | 0.87, 0.82, ~~0.35~~ | **0.90, 0.86** |
| "a remote control" | 0.95, 0.94 |由0.94, 0.94 |
| 参数 | 859.9M | 873.2M (+1.5%) |

### 跟踪器传播速度（对象多路复用）

这是 SAM 3.1 的亮点——MultiplexMaskDecoder 在单次前向传递中处理多达 16 个对象：

| 对象数 | SAM 3 | SAM 3.1 | 加速 |
|---------|-------|---------|---------|
| 3 (视频) | 547ms/帧 | 227ms/帧 | **2.4x** |
| 4 | 608ms/帧 | 203ms/帧 | **3.0x** |
| 5 | 766ms/帧 | 190ms/帧 | **4.0x** |

SAM 3 线性缩放（~150ms × N 对象）。SAM 3.1 大致恒定（~190-227ms），无论对象数量如何——一个 `track_step` 同时处理所有对象。

```
SAM 3 跟踪（4 个对象）：  4 × track_step = 4 × 150ms = 608ms/帧 (1.6 FPS)
SAM 3.1 跟踪（4 个对象）：1 × track_step =             203ms/帧 (4.9 FPS)
```

> **注意：** Meta 报告在 H100 GPU 上处理 128 个对象时约 7 倍。加速随对象数量缩放——对象越多 = SAM 3.1 的优势越大。

### 优化的实时管道

四种优化结合：骨干缓存、DETR 编码器缓存、MLX 原生后处理和快速叠加渲染。

| 分辨率 | 基线 | 优化（缓存） | 加速 |
|-----------|----------|-------------------|---------|
| 224px (2 个提示词, 5 个对象) | ~212ms (5 FPS) | **43ms (23 FPS)** | **4.6x** |
| 1008px (1 个提示词) | ~992ms (1 FPS) | **97ms 传播** | **3.9x 平均** |

224px 时的优化细分：

| 优化 | 缓存帧 | FPS | 节省 |
|-------------|-------------|-----|---------|
| 基线（无缓存） | ~212ms | ~5 | — |
| + 骨干缓存 | ~147ms | ~7 | 跳过 ViT (62ms) |
| + DETR 编码器缓存 | ~135ms | ~7.4 | 跳过编码器 (12ms) |
| + 快速叠加 | **43ms** | **23** | 跳过轮廓 (93ms) |

224px（缓存）时的每帧细分：

| 组件 | 时间 | 说明 |
|-----------|------|-------|
| DETR 解码器 + 评分 + 掩码 | 37ms | 编码器已缓存，骨干已缓存 |
| 叠加渲染 | 6ms | 布尔索引，无轮廓 |
| **总计** | **43ms** | **23 FPS** |

1008px 时的每帧细分：

| 帧类型 | 时间 | 说明 |
|-----------|------|-------|
| 检测 + ViT | 972ms | 完整骨干 + DETR（每 15 帧） |
| 传播 + ViT | 880ms | 骨干 + 跟踪器（每 5 帧） |
| 传播（缓存） | 97ms | 仅跟踫踪器，跳过 ViT（大多数帧） |

## 权重转换

SAM 3.1 权重作为 Meta `.pt` 检查点分发，而不是 HuggingFace safetensors。转换器处理：
- 键重映射 (`detector.*` → `detector_model.*`，`tracker.*` → `tracker_model.*`)
- QKV 拆分（融合的 `in_proj_weight` → 分离的 `q/k/v_proj`）
- 从位置嵌入中剔除 CLS token
- Conv2d/ConvTranspose2d 转置（PyTorch → MLX）
- 文本投影转置

```bash
python -m mlx_vlm.models.sam3_1.convert_weights --output /path/to/output
```

## 许可证

原始 SAM 3.1 模型权重由 Meta 在 [**SAM 许可证**](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE) 下发布，这是一个自定义的宽松许可证，授予非独占的、全球性的、免版税许可，以使用、复制、分发和修改 SAM 材料。关键点：

- 允许商业和研究用途
- 衍生作品必须包含 SAM 许可证的副本和对 Meta 的归属
- 按原样提供，无担保
- 受适用的贸易管制约束

使用它即表示您同意 Meta 的 SAM 许可证条款。有关详细信息，请参阅 [完整许可证文本](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE)。
