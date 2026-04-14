# PaliGemma

PaliGemma 是 Google 推出的轻量级视觉语言模型，结合 PaLI 和 Gemma 的优势。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `google/paligemma-3b` | 3B | ~8 GB | Yes |
| `google/paligemma-4b` | 4B | ~10 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model google/paligemma-3b \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/paligemma-3b")

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

- Google 官方轻量级 VLM
- SigLIP 视觉编码器
- Gemma 语言模型
- 适合边缘部署
