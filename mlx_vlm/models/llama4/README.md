# Llama 4 Vision

Llama 4 Vision 是 Meta 推出的最新视觉语言模型。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `meta-llama/Llama-4-Vision` | 需确认 | ~30 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model meta-llama/Llama-4-Vision \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("meta-llama/Llama-4-Vision")

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

- Meta 最新一代
- 性能优异
- 适合复杂任务
