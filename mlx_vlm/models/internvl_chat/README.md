# InternVL-Chat

InternVL-Chat 是 OpenGVLab 推出的视觉语言模型，在多个视觉理解任务上表现优异。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `OpenGVLab/InternVL-Chat-V1-5` | 需确认 | ~30 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model OpenGVLab/InternVL-Chat-V1-5 \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("OpenGVLab/InternVL-Chat-V1-5")

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

- 中文支持良好
- 强大的 OCR 能力
- 适合文档理解
