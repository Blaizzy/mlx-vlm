# Gemma 3N

Gemma 3N 是 Gemma 3 的变体版本，针对特定任务进行了优化。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `google/gemma-3n-4b-it` | 需确认 | ~10 GB | Yes |
| `google/gemma-3n-12b-it` | 需确认 | ~26 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model google/gemma-3n-12b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-3n-12b-it")

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

- Gemma 3 的优化版本
- 针对特定任务优化
- 详见模型文档
