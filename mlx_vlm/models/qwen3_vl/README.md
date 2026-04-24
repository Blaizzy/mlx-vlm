# Qwen3-VL

Qwen3-VL 是通义实验室最新一代视觉语言模型，在多模态理解和推理能力上有显著提升。

## 模型

| 模型 | 参数 | 显存 | 视觉 | 音频 |
|------|------|------|------|------|
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | ~10 GB | Yes | - |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | ~18 GB | Yes | - |
| `Qwen/Qwen3-VL-26B-Instruct` | 26B | ~55 GB | Yes | - |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --image path/to/image.jpg \
  --prompt "详细描述这张图片的内容" \
  --max-tokens 500
```

### 视觉推理

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --image path/to/image.jpg \
  --prompt "图片中有几个物体？分别是什么？" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen3-VL-8B-Instruct")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "分析这张图片",
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

## 架构

- **视觉编码器**: 改进的 ViT 架构
- **多尺度特征**: 支持不同分辨率输入
- **增强推理**: 更强的视觉推理能力

## 注意事项

- 推荐用于复杂视觉理解任务
- 支持中英文多语言
- 相比 Qwen2-VL 有更好的推理能力
