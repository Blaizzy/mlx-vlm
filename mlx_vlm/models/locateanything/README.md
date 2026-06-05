# LocateAnything

LocateAnything is NVIDIA's 3B vision-language grounding model for locating objects and referred regions in an image. This MLX-VLM port supports the MoonViT vision tower, Qwen2.5 text backbone, custom image processor, and the model-specific Parallel Box Decoding path.

## Model

| | |
|---|---|
| **Model ID** | `nvidia/LocateAnything-3B` |
| **Architecture** | MoonViT vision encoder + MLP connector + Qwen2.5 language model |
| **Parameters** | 3B |
| **Modalities** | Image + text |
| **Primary Tasks** | Visual grounding, open-vocabulary object localization, referring expression localization |

## CLI

The standard CLI uses autoregressive generation.

```bash
mlx_vlm.generate \
  --model nvidia/LocateAnything-3B \
  --image examples/images/cats.jpg \
  --prompt "Locate the cats." \
  --max-tokens 128 \
  --temperature 0.0
```

## Python

### Autoregressive generation

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("nvidia/LocateAnything-3B")

prompt = apply_chat_template(
    processor,
    model.config,
    "Locate the cats.",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image="examples/images/cats.jpg",
    max_tokens=128,
    temperature=0.0,
)
print(result.text)
```

### Parallel Box Decoding

LocateAnything also exposes Parallel Box Decoding through `model.pbd_generate`. Use this direct model API for the `fast`, `hybrid`, and `slow` modes.

```python
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

model, processor = load("nvidia/LocateAnything-3B")

prompt = apply_chat_template(
    processor,
    model.config,
    "Locate the cats.",
    num_images=1,
)
inputs = prepare_inputs(
    processor,
    images=["examples/images/cats.jpg"],
    prompts=prompt,
)

input_ids = inputs.pop("input_ids")
inputs.pop("attention_mask", None)

tokens = model.pbd_generate(
    input_ids,
    generation_mode="hybrid",
    max_tokens=128,
    **inputs,
)
print(processor.decode(tokens, skip_special_tokens=False))
```

`generation_mode` accepts:

- `hybrid`: starts with Parallel Box Decoding and falls back to autoregressive decoding when needed.
- `fast`: uses Parallel Box Decoding only.
- `slow`: uses autoregressive decoding through the LocateAnything PBD wrapper.

## Architecture

- **Vision tower**: MoonViT image encoder with 14x14 patches, 2D RoPE, and patch merging.
- **Connector**: LayerNorm + two linear layers projecting merged vision features into the language hidden size.
- **Language model**: Qwen2-style decoder with tied embeddings.
- **Processor**: Expands `<image-N>` placeholders into `<img>...<IMG_CONTEXT>...</img>` spans based on the processed image grid.
- **PBD**: Generates fixed-size box token blocks for the model's coordinate format.

## Folder Structure

```text
mlx_vlm/models/locateanything/
  __init__.py
  config.py
  image_processing_locateanything.py
  language.py
  locateanything.py
  pbd.py
  processing_locateanything.py
  vision.py
```

## Notes

- For multi-image prompts, pass `num_images=len(images)` and provide the same number of images to `generate` or `prepare_inputs`.
- Increase `--max-tokens` for scenes with many objects.
- The custom processor supports `save_pretrained()` and writes `processor_config.json`, `preprocessor_config.json`, and `chat_template.json`.
