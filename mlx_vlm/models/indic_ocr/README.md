# IndicOCR

OCR for English and 22 Indian languages. IndicDocLayout, a PP-DocLayoutV3
fine-tune, detects page regions and reading order; IndicBlockOCR, a
Qwen3.5-0.8B fine-tune, transcribes text, equations, and tables.

- **Original model:** [bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr)
- **MLX port:** [HashNuke/indic-ocr-mlx](https://huggingface.co/HashNuke/indic-ocr-mlx)

## Parse a page

From an mlx-vlm checkout, install with `pip install -e .`. Replace `page.png`
with an image path:

```python
from mlx_vlm.utils import get_model_path, load_model

with load_model(get_model_path("HashNuke/indic-ocr-mlx")) as parser:
    page = parser.parse("page.png")

print(page.markdown)
page.save("page.json")
```

The parser is an `nn.Module`. Its `sanitize()` routes indexed tensors to the
layout and OCR submodels. The standard loader handles weight assignment,
quantization, and evaluation. The OCR processor is loaded from the OCR stage
when `parse()` is first called.

Set parsing options on the returned parser (for example,
`parser.ocr.options = RecognizerOptions(max_tokens=128)`). To use a different
layout model, assign `parser.layout.model = load_model(layout_path)`.
`mlx_vlm.load` and `generate` operate on the OCR stage alone, loaded from the
downloaded repository's `weights/ocr` directory.

`page.markdown` contains reading-ordered text, HTML tables, and LaTeX
equations. `page.save` writes image metadata and blocks to JSON, with
pixel-coordinate `[x0, y0, x1, y1]` boxes and zero-based reading order.
