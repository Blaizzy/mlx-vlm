# PaddleOCR-VL

PaddleOCR-VL 是百度 PaddlePaddle 生态的 OCR 视觉语言模型。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `PaddlePaddle/PaddleOCR-VL` | 需确认 | ~5 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

```sh
python -m mlx_vlm.generate \
  --model PaddlePaddle/PaddleOCR-VL \
  --image path/to/text_image.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("PaddlePaddle/PaddleOCR-VL")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "Extract text",
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

- 专注 OCR 任务
- 多语言文本识别
- PaddlePaddle 生态
