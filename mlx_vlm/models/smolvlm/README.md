# SmolVLM

SmolVLM 是 Hugging Face 推出的轻量级视觉语言模型，适合在资源受限的环境中运行。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `HuggingFaceM4/SmolVLM-256M` | 256M | ~1 GB | Yes |
| `HuggingFaceM4/SmolVLM-500M` | 500M | ~2 GB | Yes |
| `HuggingFaceM4/SmolVLM-Instruct` | 500M | ~2 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 快速推理

```sh
python -m mlx_vlm.generate \
  --model HuggingFaceM4/SmolVLM-Instruct \
  --image path/to/image.jpg \
  --prompt "What is in this image?" \
  --max-tokens 300
```

### 批量处理

```sh
python -m mlx_vlm.generate \
  --model HuggingFaceM4/SmolVLM-256M \
  --image image.jpg \
  --prompt "Describe briefly" \
  --max-tokens 100
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("HuggingFaceM4/SmolVLM-Instruct")

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

- **轻量设计**: 优化参数效率
- **视觉编码器**: 简化的 SigLIP
- **语言模型**: 小型但高效的语言模型

## 注意事项

- 非常适合边缘设备部署
- 推理速度快
- 适合简单视觉任务
