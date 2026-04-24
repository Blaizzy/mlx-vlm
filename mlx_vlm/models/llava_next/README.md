# LLaVA-NeXT

LLaVA-NeXT (也称为 LLaVA-1.6) 是 LLaVA 的增强版本，提供更好的视觉理解和推理能力。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `llava-hf/llava-v1.6-vicuna-7b-hf` | 7B | ~16 GB | Yes |
| `llava-hf/llava-v1.6-vicuna-13b-hf` | 13B | ~28 GB | Yes |
| `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | ~16 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model llava-hf/llava-v1.6-vicuna-7b-hf \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("llava-hf/llava-v1.6-vicuna-7b-hf")

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

## 注意事项

- 相比 LLaVA-1.5 有显著提升
- 支持更高分辨率输入
- 更强的 OCR 能力
