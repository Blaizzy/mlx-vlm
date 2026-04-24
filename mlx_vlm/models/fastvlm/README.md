# FastVLM

FastVLM 是一个专注于推理速度的视觉语言模型，适合需要快速响应的应用场景。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `fastvlm/FastVLM` | 需确认 | ~10 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model fastvlm/FastVLM \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("fastvlm/FastVLM")

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

- 推理速度快
- 适合实时应用
- 轻量级设计
