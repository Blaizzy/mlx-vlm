# Falcon-OCR

Falcon-OCR is a 300M parameter early-fusion vision-language model from TII optimized for document OCR. It supports multiple extraction modes and integrates with a layout detector for structured document processing.

It is a single Transformer processes image patches and text tokens in a shared parameter space from the first layer, using a hybrid attention mask where image tokens attend bidirectionally and text tokens decode causally conditioned on the image.

## Model

- **Model ID**: `tiiuae/Falcon-OCR`
- **Parameters**: 300M
- **Supported categories**: `plain`, `text`, `table`, `formula`, `caption`, `footnote`, `list-item`, `page-footer`, `page-header`, `section-header`, `title`

### Links

- [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) -- code and inference engine
- [tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) -- HuggingFace model card

## When to Use What

| Mode | Best for | How |
|------|----------|-----|
| **Plain OCR** | Simple documents, real-world photos, slides, receipts, screenshots | `generate(model, processor, "plain", image=...)` |
| **Layout + OCR** | Complex multi-column documents, academic papers,reports, dense pages like newspapers | `generate_with_layout(model, processor, image=...)` |

## Installation

```bash
pip install mlx-vlm
```

Layout+OCR additionally requires PyTorch (for the layout detector):

```bash
pip install torch
```

## Python Examples

### Plain OCR
By default, category is `"plain"` (general text extraction). You can specify a category to use a task-specific prompt.
```python
from mlx_vlm import load, generate

model, processor = load("tiiuae/Falcon-OCR")

output = generate(model, processor, "plain", image="document.png", max_tokens=2000)
print(output.text)

output = generate(model, processor, "formula", image="equation.png", max_tokens=512)
print(output.text)

output = generate(model, processor, "table", image="table.png", max_tokens=512)
print(output.text)
```


### Layout + OCR (Dense Documents)

Detects document regions (text, tables, formulas, figures, etc.) using [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors), then runs OCR on each text region in reading order.

```python
from mlx_vlm import load
from mlx_vlm.models.falcon_ocr import generate_with_layout

model, processor = load("tiiuae/Falcon-OCR")

regions = generate_with_layout(model, processor, image="paper.png", max_tokens=4096)

for region in regions:
    print(f"[{region['category']}] {region.get('text', '')}")
```

Each region in the output contains:
- `category` -- the detected region type (text, table, formula, figure, etc.)
- `bbox` -- bounding box coordinates `[x1, y1, x2, y2]` in original image pixels
- `score` -- detection confidence
- `text` -- extracted text for that region
