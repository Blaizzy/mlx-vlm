# Qwen3.5

Qwen3.5 是通义实验室最新推出的语言模型，支持文本理解和生成。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `Qwen/Qwen3.5-4B` | 4B | ~10 GB | - |
| `Qwen/Qwen3.5-7B` | 7B | ~16 GB | - |
| `Qwen/Qwen3.5-72B` | 72B | ~145 GB | - |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3.5-7B \
  --prompt "你好，请介绍一下自己" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen3.5-7B")

prompt = apply_chat_template(
    processor, model.config, "你好，请介绍一下自己"
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=500,
)
print(result)
```

## 注意事项

- Qwen3 的升级版
- 纯文本模型
- 性能优异
