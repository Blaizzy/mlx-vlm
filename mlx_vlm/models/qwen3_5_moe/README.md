# Qwen3.5-MoE

Qwen3.5-MoE 是通义实验室推出的混合专家模型，通过 MoE 架构实现高效推理。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `Qwen/Qwen3.5-MoE` | 需确认 | ~20 GB | - |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3.5-MoE \
  --prompt "你好，请介绍一下自己" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen3.5-MoE")

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

- Qwen3.5 的 MoE 版本
- 推理效率高
- 适合大规模部署
