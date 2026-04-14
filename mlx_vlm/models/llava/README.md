# LLaVA

LLaVA (Large Language and Vision Assistant) 是一个开源的多模态对话模型，结合视觉编码器和语言模型实现视觉理解能力。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `llava-hf/llava-1.5-7b-hf` | 7B | ~15 GB | Yes |
| `llava-hf/llava-1.5-13b-hf` | 13B | ~28 GB | Yes |
| `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | ~16 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像描述

```sh
python -m mlx_vlm.generate \
  --model llava-hf/llava-1.5-7b-hf \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 500
```

### 视觉问答

```sh
python -m mlx_vlm.generate \
  --model llava-hf/llava-1.5-7b-hf \
  --image path/to/image.jpg \
  --prompt "What is in this image?" \
  --max-tokens 300
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("llava-hf/llava-1.5-7b-hf")

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
    max_tokens=300,
)
print(result)
```

## 架构

- **视觉编码器**: CLIP ViT-L/14
- **投影层**: 简单的线性投影
- **语言模型**: Vicuna/Llama 系列

## 注意事项

- 经典的视觉语言模型基线
- 适合一般视觉理解任务
- 社区支持广泛，文档丰富
