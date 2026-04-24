# Qwen2-VL

Qwen2-VL 是阿里巴巴通义实验室推出的开源视觉语言模型系列，支持图像理解和对话交互。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | ~5 GB | Yes |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | ~15 GB | Yes |
| `Qwen/Qwen2-VL-72B-Instruct` | 72B | ~145 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --image path/to/image.jpg \
  --prompt "描述这张图片" \
  --max-tokens 500
```

### 多图对话

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --image image1.jpg --image image2.jpg \
  --prompt "比较这两张图片的差异" \
  --max-tokens 500
```

## Python

### 基础用法

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen2-VL-7B-Instruct")

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

## 架构

- **视觉编码器**: 基于 ViT 的视觉塔
- **投影层**: 将视觉特征映射到语言空间
- **语言模型**: Qwen2 语言模型基础

## 注意事项

- 支持动态分辨率输入
- 支持多图像联合理解
- 支持中文和英文对话
