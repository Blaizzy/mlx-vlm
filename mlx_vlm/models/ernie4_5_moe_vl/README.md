# ERNIE-4.5-MoE-VL

ERNIE-4.5-MoE-VL 是百度推出的混合专家视觉语言模型，结合 MoE 架构实现高效推理。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `PaddlePaddle/ERNIE-4.5-MoE-VL` | 需确认 | ~25 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model PaddlePaddle/ERNIE-4.5-MoE-VL \
  --image path/to/image.jpg \
  --prompt "描述这张图片" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("PaddlePaddle/ERNIE-4.5-MoE-VL")

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

- 百度官方 VLM
- MoE 架构高效
- 中文支持优秀
