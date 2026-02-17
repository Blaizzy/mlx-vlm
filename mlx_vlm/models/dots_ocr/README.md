# DOTS-OCR

DOTS-OCR is a vision-language OCR model for document parsing, layout analysis, and structured extraction.

## Model

- **Primary Use Cases**: OCR, layout parsing, table extraction, formula extraction, structured JSON output

## Installation

```bash
uv pip install mlx-vlm
```

## CLI Examples

### 1) Layout JSON extraction (detailed prompt)

```bash
uv run mlx_vlm.generate --model rednote-hilab/dots.ocr --prompt "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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
  --model rednote-hilab/dots.ocr \
  --image receipt.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 1024
```

### 3) Markdown document conversion

```bash
uv run mlx_vlm.generate \
  --model mlx-community/dots.ocr-4bit \
  --image page.png \
  --prompt "Convert this page to clean Markdown while preserving reading order." \
  --max-tokens 4096
```


## Python Script Examples

### 1) `layout_json.py`

```python

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.ocr-4bit"
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

MODEL = "mlx-community/dots.ocr-4bit"
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

MODEL = "mlx-community/dots.ocr-4bit"
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

## Notes

- For long documents and layout-heavy pages, increase `--max-tokens`.
- To enforce strict structured output, keep prompts explicit about schema and sorting.
- Official DOTS README: https://huggingface.co/rednote-hilab/dots.ocr
- DOTS project/blog: https://github.com/rednote-hilab/dots.ocr
