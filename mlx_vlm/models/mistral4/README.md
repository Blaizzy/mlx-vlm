# Mistral 4 Vision

Mistral 4 Vision 是 Mistral AI 推出的最新视觉语言模型。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `mistralai/Mistral-4-Vision` | 需确认 | ~25 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model mistralai/Mistral-4-Vision \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mistralai/Mistral-4-Vision")

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

- Mistral 最新一代
- 性能提升
- 推荐使用
