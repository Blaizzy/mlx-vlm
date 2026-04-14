# Idefics3

Idefics3 是 Hugging Face 推出的最新视觉语言模型，在多个基准测试中表现优异。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `HuggingFaceM4/Idefics3-8B-Llama3` | 8B | ~18 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 基础用法

```sh
python -m mlx_vlm.generate \
  --model HuggingFaceM4/Idefics3-8B-Llama3 \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

### 多图理解

```sh
python -m mlx_vlm.generate \
  --model HuggingFaceM4/Idefics3-8B-Llama3 \
  --image img1.jpg --image img2.jpg \
  --prompt "What are the differences between these images?" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("HuggingFaceM4/Idefics3-8B-Llama3")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "Analyze this image",
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

- **视觉编码器**: SigLIP
- **语言模型**: Llama 3
- **先进训练**: 大规模视觉-语言对齐

## 注意事项

- 性能优于 Idefics2
- 支持复杂视觉推理
- 推荐用于高精度视觉理解任务
