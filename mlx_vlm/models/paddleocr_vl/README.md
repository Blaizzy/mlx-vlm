# PaddleOCR-VL

PaddleOCR-VL is a vision-language OCR model from PaddlePaddle focused on document understanding tasks such as plain OCR, table recognition, formula recognition, and chart understanding.

This MLX-VLM integration includes:
- custom processor registration for `paddleocr_vl`
- loading image geometry from `preprocessor_config.json`
- prompt formatting through `apply_chat_template`

## Model

- Hugging Face ID: `PaddlePaddle/PaddleOCR-VL`
- Best for: OCR, tables, formulas, charts, and structured document extraction

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Basic OCR

```sh
uv run mlx_vlm.generate \
  --model PaddlePaddle/PaddleOCR-VL \
  --image /path/to/document.png \
  --prompt "OCR:" \
  --max-tokens 512 \
  --temperature 0
```

### Table recognition

```sh
uv run mlx_vlm.generate \
  --model PaddlePaddle/PaddleOCR-VL \
  --image /path/to/table.png \
  --prompt "Table Recognition:" \
  --max-tokens 1024 \
  --temperature 0
```

### Formula recognition

```sh
uv run mlx_vlm.generate \
  --model PaddlePaddle/PaddleOCR-VL \
  --image /path/to/formula.png \
  --prompt "Formula Recognition:" \
  --max-tokens 512 \
  --temperature 0
```

### Chart understanding

```sh
uv run mlx_vlm.generate \
  --model PaddlePaddle/PaddleOCR-VL \
  --image /path/to/chart.png \
  --prompt "Chart Recognition:" \
  --max-tokens 1024 \
  --temperature 0
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("PaddlePaddle/PaddleOCR-VL")

image = ["/path/to/document.png"]
prompt = "OCR:"

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image,
    max_tokens=512,
    temperature=0.0,
)
print(result.text)
```

## Prompt Notes

- `OCR:` is the default prompt for plain text extraction.
- Other common task prompts are `Table Recognition:`, `Formula Recognition:`, and `Chart Recognition:`.
- For structured extraction, you can provide a task-specific instruction or schema in the prompt.

## Notes

- You usually should not manually add image placeholder tokens when using `apply_chat_template`.
- Local image paths and image URLs can both be used as inputs.
