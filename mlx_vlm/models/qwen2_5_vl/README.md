# Qwen2.5-VL

Qwen2.5-VL 是 Qwen2-VL 的升级版本，在性能和效率上都有提升。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | ~8 GB | Yes |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | ~16 GB | Yes |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 72B | ~145 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image path/to/image.jpg \
  --prompt "描述这张图片" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen2.5-VL-7B-Instruct")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "描述这张图片",
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

- Qwen2-VL 的升级版
- 性能提升
- 推荐新项目使用
