# LFM2-VL

LFM2-VL 是一个专注于长上下文理解的视觉语言模型。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `lfm2/LFM2-VL` | 需确认 | ~15 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model lfm2/LFM2-VL \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("lfm2/LFM2-VL")

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

- 长上下文理解
- 适合文档分析
