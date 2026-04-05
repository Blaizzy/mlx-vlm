# MolmoPoint

MolmoPoint 是一个具有像素级精确指向功能的视觉语言模型。给定图像和文本提示词如 "Point to the cats"，它为请求的对象生成精确的 (x, y) 坐标。

## 模型

| | |
|---|---|
| **模型 ID** | `allenai/MolmoPoint-8B` |
| **架构** | SigLIP ViT (视觉) + 注意力池化连接器 + Qwen2 样式解码器（语言）+ PointPredictor |
| **参数** | ~8B |
| **视觉编码器** | 27 层（截断到 25），1152 维，16 个头，头维度 72，patch 大小 14，输入 378x378 |
| **语言模型** | 36 层，4096 维，32 个头，8 个 KV 头，头维度 128，SwiGLU MLP |
| **连接器** | 交叉注意力池化（无输出投影）+ 门控 SiLU MLP (1152 -> 12288 -> 4096) |
| **点预测器** | 3 阶段：patch 选择（RoPE 键）-> 子patch 选择 -> 3x3 位置网格 |
| **任务** | 视觉指向/grounding、图像描述、视觉问答 |


## 指向如何工作

MolmoPoint 扩展了标准词汇表，包含 **patch**、**subpatch** 和 **location** token：

1. **Patch 选择** -- 模型选择包含目标对象的图像 patch。Patch 键从图像 token 隐藏状态构建，带有 1D 旋转嵌入。可学习的"没有更多点"类别信号指向结束。
2. **Subpatch 选择** -- 在选择的 patch 内，模型使用原始 ViT 特征（在池化之前）选择特定的 ViT 子patch。
3. **位置细化** -- 3x3 网格在子patch 内细化点，以获得子patch 级精度。

每个点编码为三元组 `<POINT_patch> <POINT_subpatch> <POINT_location> object_id`。

## CLI 使用

### 图像描述

```bash
python -m mlx_vlm.generate \
    --model allenai/MolmoPoint-8B \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200
```

### 指向对象

```bash
python -m mlx_vlm.generate \
    --model allenai/MolmoPoint-8B \
    --image path/to/image.jpg \
    --prompt "Point to the cats" \
    --max-tokens 50
```

## Python 使用

### 基本生成

```python
from mlx_vlm import load, generate

model, processor = load("allenai/MolmoPoint-8B")

output = generate(
    model,
    processor,
    "Describe this image",
    ["path/to/image.jpg"],
    max_tokens=200,
)
print(output)
```

### 点提取和可视化

```python
import mlx.core as mx
from mlx_vlm.utils import load, prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.generate import generate_step
from mlx_vlm.models.molmo_point.point_utils import (
    extract_points_from_text,
    draw_points_on_image,
)

model, processor = load("allenai/MolmoPoint-8B")
mx.eval(model.parameters())

image_path = "path/to/image.jpg"
prompt = apply_chat_template(
    processor, model.config, "Point to the cats", num_images=1
)
inputs = prepare_inputs(processor, images=image_path, prompts=prompt)

input_ids = inputs["input_ids"]
pixel_values = inputs.get("pixel_values", None)
mask = inputs.get("attention_mask", None)
kwargs = {
    k: v
    for k, v in inputs.items()
    if k not in ["input_ids", "pixel_values", "attention_mask"]
}

# 生成
tokens = []
for n, (token, _) in enumerate(
    generate_step(input_ids, model, pixel_values, mask, max_tokens=50, **kwargs)
):
    tokens.append(token)
    if n >= 49:
        break

output_text = processor.tokenizer.decode(tokens)
print(output_text)

# 从生成的文本中提取点
if hasattr(processor, "_pointing_metadata") and processor._pointing_metadata:
    points = extract_points_from_text(
        output_text,
        processor._pointing_metadata,
        no_more_points_class=model.config.no_more_points_class,
        patch_location=model.config.patch_location,
    )
    for obj_id, img_num, x, y in points:
        print(f"Object {obj_id}: ({x:.1f}, {y:.1f})")

    # 保存带注释的图像
    draw_points_on_image(image_path, points, "output_pointed.jpg")
```
## 文件夹结构

```
mlx_vlm/models/molmo_point/
    __init__.py                  # 模块导出
    config.py                    # ModelConfig, TextConfig, VisionConfig, AdapterConfig
    vision.py                    # SigLIP ViT 编码器（线性 patch 嵌入，LayerNorm）
    language.py                  # Qwen2 样式 LLM 解码器（RMSNorm, GQA, QK-norm, SwiGLU）
    molmo_point.py               # 主模型：连接器、点预测器、扩展的 LM 头，
                                 #   logit 处理器、图像缓存、生成逻辑
    image_processing.py          # 图像处理器（重叠裁剪、平铺、池化；无 torch）
    processing_molmo_point.py    # Tokenizer + 图像 token 插入
    point_utils.py               # 从 <POINT_X> token 提取点 + 可视化
```

## 架构细节

- **图像处理**：图像分解为重叠裁剪（最多 8 个，重叠边距 [4, 4]）在 378x378，加上全局调整大小的视图。每个裁剪被 patch 化为 27x27 = 729 个 14x14x3 = 588 像素的 patch。来自 ViT 层 18 和 24 的 patch 被连接（2304 维特征）。
- **连接器**：注意力池化特征（平均查询，无输出投影的交叉注意力）通过门控 SiLU MLP 投影到 LLM 维度（4096）。
- **LLM**：Qwen2 样式解码器，具有融合 QKV 投影，Q/K 的每头 RMSNorm（Qwen3 样式），以及 SwiGLU MLP。RoPE，theta=1M。
- **扩展词汇表**：标准词汇表（151,936 + 128 个额外）在运行时使用 patch、subpatch 和 location token 扩展。Logit 处理器强制有效的生成顺序（patch -> subpatch -> location）。

## 说明

- 自定义图像处理器仅使用 PIL 和 numpy（无 torch/torchvision 依赖）。
- 点提取需要调用 `prepare_inputs` 后存储在处理器上的 `_pointing_metadata`。
- 完整 bf16 模型的峰值内存约为 39 GB。
- 在 Apple Silicon（M 系列）上的生成速度约为 ~6 tokens/秒。
