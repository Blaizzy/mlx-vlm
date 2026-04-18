# Moondream3

Moondream3 是一个具有 9.27B 总参数和每个 token ~2B 活跃参数的混合专家（MoE）视觉语言模型。它结合了基于 SigLIP 的视觉编码器与 MoE 文本解码器，以实现高效的多模态理解。

## 模型

| | |
|---|---|
| **模型 ID** | `moondream/moondream3-preview` |
| **架构** | SigLIP ViT (视觉) + MoE Transformer 解码器（语言）|
| **总参数** | ~9.27B (每个 token 2B 活跃) |
| **视觉编码器** | 27 层，1152 维，16 个头，patch 大小 14，裁剪大小 378 |
| **语言模型** | 24 层，2048 维，32 个头（层 0-3 密集，4-23 MoE，64 个专家，top-8）|
| **任务** | 视觉问答、图像描述、视觉推理 |

## CLI 使用

```bash
python -m mlx_vlm.generate \
    --model moondream/moondream3-preview \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200
```

使用自定义参数：

```bash
python -m mlx_vlm.generate \
    --model moondream/moondream3-preview \
    --image path/to/image.jpg \
    --prompt "How many objects are in this image?" \
    --max-tokens 100 \
    --temp 0.0
```

## Python 使用

```python
from mlx_vlm import load, generate

model, processor = load("moondream/moondream3-preview")

output = generate(
    model,
    processor,
    "Describe this image",
    ["path/to/image.jpg"],
    max_tokens=200,
)
print(output)
```

## 架构

- **视觉**：基于 SigLIP 的 ViT 编码器，支持多裁剪（最多 12 个裁剪）。图像被 patch 化（378x378 裁剪上的 14x14 patch），每个裁剪产生 729 个 token。全局和局部裁剪特征被重构并通过 2 层 MLP 投影到 2048 维。
- **语言**：MoE Transformer 解码器，具有并行残差连接（`x = x + attn(ln(x)) + mlp(ln(x))`）。层 0-3 使用密集 MLP；层 4-23 使用混合专家，64 个专家和 top-8 路由，使用 GeGLU 激活。
- **Tau 缩放**：在注意力的 Q 和 V 上学习的位置和数据依赖温度缩放，Moondream3 独有。
- **RoPE**：旋转位置嵌入应用于 64 个头维度中的前 32 个，基数 theta=1.5M。
- **前缀注意力**：在预填充期间，前 730 个 token（1 BOS + 729 个视觉 token）的双向注意力。

## 说明

- 使用来自 `moondream/starmie-v1` 的自定义 tokenizer（SuperBPE）。
- 模型可能在答案之前输出思考 token（`<|md_reserved_4|>`）。
- 多裁剪处理通过重叠边距修剪和自适应平均池化重构局部特征。
- 完整 bf16 模型的峰值内存使用约为 24 GB。
