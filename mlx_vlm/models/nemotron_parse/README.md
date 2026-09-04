# Nemotron-Parse

Nemotron-Parse is NVIDIA's document OCR / parsing model: a C-RADIO vision encoder (ViT-Huge) paired with a compact mBART-style decoder that emits markdown annotated with bounding boxes and semantic classes. It is the first encoder-decoder OCR architecture in mlx-vlm since `florence2`.

## Model

- **Model IDs**: [`mlx-community/Nemotron-Parse-2.0-8bit`](https://huggingface.co/mlx-community/Nemotron-Parse-2.0-8bit) (also `-4bit`), [`mlx-community/Nemotron-Parse-v1.2-8bit`](https://huggingface.co/mlx-community/Nemotron-Parse-v1.2-8bit) (also `-4bit`)
- **Architecture**: C-RADIO ViT-H vision encoder (32 pre-norm blocks, 8 prefix tokens, compression neck) + 10-layer pre-norm mBART decoder with cross-attention
- **Source**: [nvidia/NVIDIA-Nemotron-Parse-2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0), [nvidia/NVIDIA-Nemotron-Parse-v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)

## Installation

```sh
pip install mlx-vlm
```

## Usage

The decoder is seeded with the tokenized prompt (matching the HF reference), so the task prompt controls what gets parsed out — it is not decorative.

### CLI

```bash
uv run mlx_vlm generate \
  --model mlx-community/Nemotron-Parse-2.0-8bit \
  --image document.png \
  --prompt "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>" \
  --max-tokens 4096
```

### Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Nemotron-Parse-2.0-8bit")

task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
formatted_prompt = apply_chat_template(processor, model.config, task_prompt, num_images=1)

output = generate(model, processor, formatted_prompt, image=["document.png"], max_tokens=4096)
print(output.text)
```

## Notes

- NVIDIA's default task prompt above requests markdown output with bounding boxes and semantic classes. Swap `predict_no_text_in_pic` for `predict_text_in_pic` if the document has embedded image text you also want transcribed. Omitting the task prompt entirely makes the decoder collapse into a repetition loop — always pass one.
- Both the tied-embedding 2.0 checkpoint and the untied-`lm_head` v1.x checkpoints are supported; `sanitize()` detects which layout a given checkpoint uses.
- Native resolution is 2048x1664; the processor resizes with aspect ratio preserved and white-pads to that size.
- Verified byte-identical against the Hugging Face CPU reference. NVIDIA's published golden generation is captured on CUDA and is not reproducible cross-hardware past the third decoder step — that is a property of the model, not a port bug.
