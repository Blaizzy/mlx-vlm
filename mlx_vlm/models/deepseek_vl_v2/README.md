# DeepSeek-VL-V2

DeepSeek-VL-V2 是 DeepSeek 推出的第二代视觉语言模型，在视觉理解和推理能力上有显著提升。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `deepseek-ai/deepseek-vl-v2` | 需确认 | ~20 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model deepseek-ai/deepseek-vl-v2 \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("deepseek-ai/deepseek-vl-v2")

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

- DeepSeek 第二代 VLM
- 强大的推理能力
- 中英文支持
