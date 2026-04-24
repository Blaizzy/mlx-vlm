# Jina-VLM

Jina-VLM 是 Jina AI 推出的视觉语言模型，专注于高效的视觉理解。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `jinaai/jina-vlm` | 需确认 | ~10 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model jinaai/jina-vlm \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("jinaai/jina-vlm")

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

- Jina AI 官方模型
- 高效推理
- 适合嵌入式场景
