# Molmo 2

Molmo 2 是 Allen AI 推出的 Molmo 第二代模型，性能和效率都有显著提升。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `allenai/Molmo-7B-O-v2` | 7B | ~16 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model allenai/Molmo-7B-O-v2 \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("allenai/Molmo-7B-O-v2")

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

- Molmo 的升级版本
- 更好的性能
- 保持可解释性
