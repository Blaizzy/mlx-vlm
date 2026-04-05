# Granite Vision 3.2

IBM 的 Granite Vision 3.2 模型，用于文档理解和视觉问答。结合了 SigLIP 视觉编码器与 Granite 语言模型，使用 LLaVA-NeXT AnyRes 架构。

## 模型

| | |
|---|---|
| **模型 ID** | `mlx-community/granite-vision-3.2-2b-bf16` |
| **架构** | Granite LM + SigLIP (视觉) + MLP 投影器 (LLaVA-NeXT) |
| **参数** | ~3B |
| **视觉编码器** | SigLIP (27 层, 1152 hidden, 384px, patch 14) |
| **视觉特征** | 来自层 [-24, -20, -12, -1] 的多层连接 |
| **任务** | 文档理解、VQA、图像描述、表格/图表分析 |

## CLI 使用

```bash
python -m mlx_vlm.generate \
    --model mlx-community/granite-vision-3.2-2b-bf16 \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

## Python 使用

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/granite-vision-3.2-2b-bf16")

image = "path/to/image.jpg"
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."}
    ]}
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

output = generate(model, processor, image=image, prompt=prompt, max_tokens=512)
print(output)
```

## 架构

- **视觉**：SigLIP 编码器（27 transformer 层，无 CLS token）
- **语言**：Granite 3.1-2B，具有 MUP 乘数（嵌入、注意力、残差、logits 缩放）
- **投影器**：多层特征连接（4 层 → 4608-dim）→ 2 层 MLP → 2048-dim
- **AnyRes**：通过图像网格固定点进行可变分辨率（27 种配置）
