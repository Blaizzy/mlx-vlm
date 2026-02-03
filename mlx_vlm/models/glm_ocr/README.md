# GLM-OCR

GLM-OCR is a vision-language model optimized for Optical Character Recognition (OCR) tasks. It excels at document parsing and structured information extraction.

## Model

- **Model ID**: `mlx-community/GLM-OCR-bf16`
- **Architecture**: Vision-Language Model with M-RoPE (Multi-dimensional Rotary Position Embedding)
- **Supported Tasks**: Text recognition, formula recognition, table recognition, and structured information extraction

## Installation

**Install with pip:**
```sh
pip install mlx-vlm
```

**Or with uv (fast Python package manager):**
```sh
uv pip install mlx-vlm
```

## Usage

### CLI

**Basic text recognition:**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image document.png --prompt "Text Recognition:"
```

**Formula recognition:**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image equation.png --prompt "Formula Recognition:"
```

**Table recognition:**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image table.png --prompt "Table Recognition:"
```

**Structured information extraction:**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image id_card.png --prompt '请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}'
```

### Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Load model
model, processor = load("mlx-community/GLM-OCR-bf16")

# Document Parsing - Text Recognition
prompt = "Text Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["document.png"],
    max_tokens=512,
    verbose=True,
)
print(result.text)
```

**Formula Recognition:**
```python
prompt = "Formula Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["equation.png"],
    max_tokens=256,
)
print(result.text)
```

**Table Recognition:**
```python
prompt = "Table Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["table.png"],
    max_tokens=1024,
)
print(result.text)
```

**Structured Information Extraction:**
```python
# Define JSON schema for extraction
prompt = """请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}"""

formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["id_card.png"],
    max_tokens=512,
)
print(result.text)
```

## Supported Prompts

GLM-OCR supports two types of prompt scenarios:

### 1. Document Parsing

Extract raw content from documents using these task prompts:

| Task | Prompt |
|------|--------|
| Text Recognition | `Text Recognition:` |
| Formula Recognition | `Formula Recognition:` |
| Table Recognition | `Table Recognition:` |

### 2. Information Extraction

Extract structured information from documents. Prompts must follow a strict JSON schema format.

**Example - ID Card Extraction:**
```
请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}
```

**Example - Invoice Extraction:**
```
请按下列JSON格式输出图中信息:
{
    "invoice_number": "",
    "date": "",
    "vendor": "",
    "items": [],
    "subtotal": "",
    "tax": "",
    "total": ""
}
```

> **Note**: When using information extraction, the output must strictly adhere to the defined JSON schema to ensure downstream processing compatibility.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_tokens` | Maximum number of tokens to generate | 256 |
| `temperature` | Sampling temperature (0 = deterministic) | 0.0 |
| `top_p` | Nucleus sampling parameter | 1.0 |

## Example: Complete OCR Pipeline

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import json

# Load model once
model, processor = load("mlx-community/GLM-OCR-bf16")

def extract_text(image_path: str) -> str:
    """Extract raw text from an image."""
    prompt = "Text Recognition:"
    formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
    result = generate(model, processor, formatted_prompt, image=[image_path], max_tokens=1024)
    return result.text

def extract_structured(image_path: str, schema: dict) -> dict:
    """Extract structured information using a JSON schema."""
    prompt = f"请按下列JSON格式输出图中信息:\n{json.dumps(schema, indent=4, ensure_ascii=False)}"
    formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
    result = generate(model, processor, formatted_prompt, image=[image_path], max_tokens=512)
    return json.loads(result.text)

# Usage
text = extract_text("document.png")
print(f"Extracted text: {text}")

# Structured extraction
schema = {
    "title": "",
    "author": "",
    "date": "",
    "content": ""
}
data = extract_structured("article.png", schema)
print(f"Extracted data: {data}")
```

## Acknowledgements

This model is a port of **[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR))** developed by the [ZAI Team](https://huggingface.co/zai-org). We thank the ZAI team for their work on this powerful OCR model.
