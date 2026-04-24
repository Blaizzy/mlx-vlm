# Kimi-VL

Kimi-VL 是 Moonshot AI 推出的视觉语言模型，支持图像理解和多模态对话。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `moonshot/Kimi-VL` | 需确认 | ~20 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model moonshot/Kimi-VL \
  --image path/to/image.jpg \
  --prompt "描述这张图片" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("moonshot/Kimi-VL")

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

- Moonshot AI 官方模型
- 中文支持优秀
- 长文本处理能力
