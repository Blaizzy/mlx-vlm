# Gemma 3

Gemma 3 是 Google 推出的视觉语言模型系列，结合先进的语言能力和视觉理解。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `google/gemma-3-4b-it` | 4B | ~10 GB | Yes |
| `google/gemma-3-12b-it` | 12B | ~26 GB | Yes |
| `google/gemma-3-27b-it` | 27B | ~55 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model google/gemma-3-12b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-3-12b-it")

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

- Gemma 4 的前代版本
- 优秀的对话能力
- 适合多轮对话场景
