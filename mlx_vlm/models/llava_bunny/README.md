# LLaVA-Bunny

LLaVA-Bunny 是 LLaVA 的轻量级变体，专注于效率和小型化。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `bunny/LLaVA-Bunny` | 需确认 | ~5 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model bunny/LLaVA-Bunny \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 300
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("bunny/LLaVA-Bunny")

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
    max_tokens=300,
)
print(result)
```

## 注意事项

- 轻量级设计
- 适合资源受限环境
- 快速推理
