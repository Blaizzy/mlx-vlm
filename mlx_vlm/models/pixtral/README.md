# Pixtral

Pixtral 是 Mistral AI 推出的视觉语言模型，专注于图像理解和多模态交互。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `mistralai/Pixtral-12B` | 12B | ~26 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model mistralai/Pixtral-12B \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 500
```

### OCR 任务

```sh
python -m mlx_vlm.generate \
  --model mistralai/Pixtral-12B \
  --image path/to/text_image.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mistralai/Pixtral-12B")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "What do you see?",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=image,
    max_tokens=500,
)
print(result)
```

## 架构

- **视觉编码器**: Mistral 专用视觉塔
- **语言模型**: Mistral 基础
- **端到端优化**: 统一训练框架

## 注意事项

- Mistral AI 官方 VLM
- 强大的 OCR 能力
- 适合文档理解和图像分析
