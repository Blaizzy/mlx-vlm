# GOT-OCR 2.0 (General OCR Theory 2.0)

GOT-OCR 2.0 is an 8.3B parameter vision-language model developed by StepFun for unified OCR tasks, including document OCR, formatted scene text recognition, table and formula parsing, chart processing, and fine-grained visual text grounding.

## Model Overview

- **Model IDs**:
  - [`stepfun-ai/GOT-OCR2_0`](https://huggingface.co/stepfun-ai/GOT-OCR2_0) (Upstream)
  - [`mlx-community/GOT-OCR2_0-bf16`](https://huggingface.co/mlx-community/GOT-OCR2_0-bf16)
  - [`mlx-community/GOT-OCR2_0-4bit`](https://huggingface.co/mlx-community/GOT-OCR2_0-4bit)
  - [`mlx-community/GOT-OCR2_0-8bit`](https://huggingface.co/mlx-community/GOT-OCR2_0-8bit)
- **Quantizations Collection**: [axiom-of-choice/mlx-quantizations](https://huggingface.co/collections/axiom-of-choice/mlx-quantizations)
- **Architecture**: ViT-based `ImageEncoderViT` (1024x1024 input downsampled to 256 visual tokens) with window attention and decomposed relative positional embeddings + linear projector + Qwen2-based language decoder.
- **Source**: [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0)

## Installation

```bash
pip install mlx-vlm
```

or with `uv`:

```bash
uv pip install mlx-vlm
```

## Usage

### CLI

**Plain Text OCR:**
```bash
mlx_vlm generate \
  --model mlx-community/GOT-OCR2_0-bf16 \
  --image document.png \
  --prompt "OCR:" \
  --max-tokens 1024
```

**Formatted Document OCR (Markdown):**
```bash
mlx_vlm generate \
  --model mlx-community/GOT-OCR2_0-bf16 \
  --image document.png \
  --prompt "format:" \
  --max-tokens 2048
```

**Fine-grained / Crop Region OCR:**
```bash
mlx_vlm generate \
  --model mlx-community/GOT-OCR2_0-bf16 \
  --image document.png \
  --prompt 'OCR with format: {"box": [100, 100, 500, 500]}' \
  --max-tokens 1024
```

### Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Load model
model, processor = load("mlx-community/GOT-OCR2_0-bf16")

# Document OCR (formatted markdown output)
prompt = "format:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

output = generate(
    model,
    processor,
    formatted_prompt,
    image="document.png",
    max_tokens=2048,
    temperature=0.0,
)
print(output.text)
```

## Prompting Modes

GOT-OCR 2.0 supports several task modes specified in the prompt:

| Mode | Prompt | Description |
|---|---|---|
| **Plain Text OCR** | `OCR:` | Standard text extraction without special markdown layout. |
| **Formatted OCR** | `format:` | Extracts text with layout preservation (markdown tables, headers, formulas). |
| **Fine-grained Box OCR** | `OCR with format: {"box": [y1, x1, y2, x2]}` | OCR restricted to a normalized bounding box range `[0, 1000]`. |
| **Fine-grained Color OCR** | `OCR with format: {"color": "red"}` | OCR restricted to text rendered in a specific color. |

## Technical Details

- **Resolution & Padding**: Input images are resized to `1024x1024` using bicubic interpolation.
- **Visual Tokens**: The vision encoder produces 256 image tokens wrapped in `<img><imgpad>...<imgpad></img>`.
- **Conversation Template**: Prompts are automatically wrapped into the MPT conversation template (`<|im_start|>system...<|im_end|>`) used during training.
- **Stop Tokens**: Stop token IDs include both `<|im_end|>` (`151645`) and `<|endoftext|>` (`151643`).
