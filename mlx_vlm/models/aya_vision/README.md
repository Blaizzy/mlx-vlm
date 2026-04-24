# Aya-Vision

Aya-Vision 是 Cohere 推出的多语言视觉语言模型，支持多种语言的视觉理解。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `cohere/Aya-Vision` | 需确认 | ~15 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model cohere/Aya-Vision \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("cohere/Aya-Vision")

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

## 注意事项

- 多语言支持优秀
- 适合国际化应用
