# Idefics2

Idefics2 是 Hugging Face 推出的视觉语言模型，支持图像理解和多模态对话。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `HuggingFaceM4/Idefics2-8b` | 8B | ~18 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model HuggingFaceM4/Idefics2-8b \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("HuggingFaceM4/Idefics2-8b")

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

- Idefics3 的前代版本
- 仍适合生产环境使用
- 社区支持良好
