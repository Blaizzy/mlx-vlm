# Phi-3-V

Phi-3-V 是微软推出的轻量级视觉语言模型，在保持高效的同时提供强大的视觉理解能力。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `microsoft/Phi-3-vision-4b` | 4B | ~10 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model microsoft/Phi-3-vision-4b \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("microsoft/Phi-3-vision-4b")

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

- 轻量级设计
- 推理速度快
- 适合资源受限环境
