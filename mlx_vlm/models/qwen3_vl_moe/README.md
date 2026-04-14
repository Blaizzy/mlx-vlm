# Qwen3-VL-MoE

Qwen3-VL-MoE 是通义实验室推出的视觉语言混合专家模型。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `Qwen/Qwen3-VL-MoE` | 需确认 | ~25 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model Qwen/Qwen3-VL-MoE \
  --image path/to/image.jpg \
  --prompt "详细描述这张图片的内容" \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Qwen/Qwen3-VL-MoE")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "详细描述这张图片的内容",
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

- Qwen3-VL 的 MoE 版本
- 推理效率更高
- 适合大规模部署
