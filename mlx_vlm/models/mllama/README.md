# MLLaMA

MLLaMA (Multimodal LLaMA) 是 Meta 推出的多模态版本，结合 LLaMA 语言模型和视觉理解能力。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | 11B | ~24 GB | Yes |
| `meta-llama/Llama-3.2-90B-Vision-Instruct` | 90B | ~180 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

### 复杂推理

```sh
python -m mlx_vlm.generate \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --image path/to/image.jpg \
  --prompt "What is the mood of this image? Explain why." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("meta-llama/Llama-3.2-11B-Vision-Instruct")

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

- **视觉编码器**: 专门的视觉塔
- **语言模型**: Llama 3.2 架构
- **原生多模态**: 端到端训练

## 注意事项

- Meta 官方多模态模型
- 优秀的对话能力
- 适合复杂视觉推理
