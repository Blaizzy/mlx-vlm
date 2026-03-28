# DOTS OCR

DOTS models are vision-language OCR models for document parsing, layout analysis, and structured extraction. `dots.mocr` extends the original `dots.ocr` checkpoint with stronger multilingual parsing and structured-graphics generation capabilities.

## Model

- **Primary Use Cases**: OCR, layout parsing, table extraction, formula extraction, structured JSON output

## Installation

```bash
uv pip install mlx-vlm
```

## CLI Examples

### 1) Layout JSON extraction (detailed prompt)

```bash
uv run mlx_vlm.generate --model rednote-hilab/dots.mocr --prompt "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object." --image "path_to_image.jpg" --max-tokens 5000
```

### 2) Basic OCR

```bash
uv run mlx_vlm.generate \
  --model rednote-hilab/dots.mocr \
  --image receipt.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 1024
```

### 3) Markdown document conversion

```bash
uv run mlx_vlm.generate \
  --model mlx-community/dots.mocr-4bit \
  --image page.png \
  --prompt "Convert this page to clean Markdown while preserving reading order." \
  --max-tokens 4096
```


## Python Script Examples

### 1) `layout_json.py`

```python

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "path_to_image.jpg"

PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=5000,
    temperature=0.0,
)
print(result.text)
```

### 2) `basic_ocr.py`

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "receipt.jpg"
PROMPT = "Extract all text from this image."

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=1024,
    temperature=0.0,
)
print(result.text)
```

### 3) `markdown_document_conversion.py`
```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "page.png"
PROMPT = "Convert this page to clean Markdown while preserving reading order."

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=4096,
    temperature=0.0,
)
print(result.text)
```

## Notebook Walkthrough

- [`examples/dots_mocr_demo.ipynb`](../../../examples/dots_mocr_demo.ipynb) runs the unique upstream `dots.mocr` README scenarios with MLX-VLM only, uses bundled local demo assets, and saves raw outputs, overlays, rendered SVG previews, and a contact sheet. Invalid SVG generations fall back to a readable text overlay instead of a renderer error page.
- The bundled source assets for that notebook live under [`examples/images`](../../../examples/images).
- The notebook writes alias artifacts for `demo_hf_layout` and `parser_image_default` because those upstream examples reuse the same image and prompt as the main document-parsing run.
- SVG preview rendering in the notebook uses macOS `qlmanage`.

## Notes

- For long documents and layout-heavy pages, increase `--max-tokens`.
- To enforce strict structured output, keep prompts explicit about schema and sorting.
- Official `dots.ocr` README: https://huggingface.co/rednote-hilab/dots.ocr
- Official `dots.mocr` README: https://huggingface.co/rednote-hilab/dots.mocr
- DOTS project/blog: https://github.com/rednote-hilab/dots.mocr
