# Qwen3-Omni-MoE

Qwen3-Omni-MoE 是通义实验室推出的全模态混合专家模型，支持文本、图像、音频等多种输入。

## 模型

| 模型 | 参数 | 显存 | 视觉 | 音频 |
|------|------|------|------|------|
| `Qwen/Qwen3-Omni-MoE` | 需确认 | ~25 GB | Yes | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3-Omni-MoE \
  --image path/to/image.jpg \
  --prompt "描述这张图片" \
  --max-tokens 500
```

### 音频理解

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3-Omni-MoE \
  --audio path/to/audio.wav \
  --prompt "转录这段音频" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen3-Omni-MoE")

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

- 全模态支持
- MoE 架构高效
- 适合复杂多模态任务
