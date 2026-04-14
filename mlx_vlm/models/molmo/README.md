# Molmo

Molmo (也称为 OLMo) 是 Allen AI 推出的开源视觉语言模型，专注于可解释性和研究。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `allenai/Molmo-7B-D` | 7B | ~16 GB | Yes |
| `allenai/Molmo-7B-O` | 7B | ~16 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model allenai/Molmo-7B-D \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("allenai/Molmo-7B-D")

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

- 专注于可解释性
- 研究友好设计
- 完全开源权重
