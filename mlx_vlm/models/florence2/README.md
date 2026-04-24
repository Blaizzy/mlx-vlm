# Florence2

Florence2 是微软推出的视觉语言模型，专注于 OCR 和文档理解任务。

## 模型

| 模型 | 参数 | 显存 | 视觉 |
|------|------|------|------|
| `microsoft/Florence-2-base` | 需确认 | ~3 GB | Yes |
| `microsoft/Florence-2-large` | 需确认 | ~8 GB | Yes |

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### OCR 任务

```sh
python -m mlx_vlm.generate \
  --model microsoft/Florence-2-base \
  --image path/to/document.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 500
```

### 文档理解

```sh
python -m mlx_vlm.generate \
  --model microsoft/Florence-2-large \
  --image path/to/document.jpg \
  --prompt "Summarize this document." \
  --max-tokens 500
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("microsoft/Florence-2-base")

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

- 专注 OCR 和文档理解
- 多语言文本识别
- 适合文档处理应用
