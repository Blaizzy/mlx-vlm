# Mage-Flow

Mage-Flow is Microsoft's 4B native-resolution image generation and editing
family. `mlx-vlm` loads the original Hugging Face Diffusers-style checkpoints
directly and runs the Qwen3-VL conditioner, NR-MMDiT transformer, Mage-VAE, and
flow-matching sampler with MLX.

## Models

| Hugging Face model | Alias | Task | Recommended settings |
|---|---|---|---|
| `microsoft/Mage-Flow-Base` | `mage-flow-base` | Generation | 30 steps, guidance 5 |
| `microsoft/Mage-Flow` | `mage-flow` | Generation | 20 steps, guidance 5 |
| `microsoft/Mage-Flow-Turbo` | `mage-flow-turbo` | Generation | 4 steps, guidance 1 |
| `microsoft/Mage-Flow-Edit-Base` | `mage-flow-edit-base` | Editing | 30 steps, guidance 5 |
| `microsoft/Mage-Flow-Edit` | `mage-flow-edit` | Editing | 30 steps, guidance 5 |
| `microsoft/Mage-Flow-Edit-Turbo` | `mage-flow-edit-turbo` | Editing | 4 steps, guidance 1 |

All variants support native resolutions from 512 to 2048 pixels per side,
including aspect ratios up to 4:1. Width and height must be multiples of 16.

The first load downloads the checkpoint's transformer, VAE, text encoder,
tokenizer, and scheduler metadata. The text encoder is evicted after prompt
encoding by default to reduce peak resident memory before denoising.

## CLI

Generate with the aligned model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task generate \
  --model microsoft/Mage-Flow \
  --prompt "A tiny glass greenhouse on a mossy forest floor at sunrise" \
  --size 1024x1024 \
  --steps 20 \
  --guidance 5 \
  --seed 42 \
  --output outputs/mage-flow.png
```

Generate with the four-step Turbo model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model mage-flow-turbo \
  --prompt "Editorial photograph of a cobalt teapot on a red table" \
  --size 1024x1024 \
  --steps 4 \
  --guidance 1 \
  --output outputs/mage-flow-turbo.png
```

Edit one image:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task edit \
  --model microsoft/Mage-Flow-Edit \
  --image input/dog.jpg \
  --prompt "Replace the background with a field of sunflowers" \
  --size 1024x1024 \
  --steps 30 \
  --guidance 5 \
  --output outputs/mage-flow-edit.png
```

`--image` accepts multiple paths for multi-reference editing:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task edit \
  --model mage-flow-edit-turbo \
  --image input/scene.png input/object.png \
  --prompt "Blend the object from image 2 naturally into image 1" \
  --size 1024x1024 \
  --steps 4 \
  --guidance 1 \
  --output outputs/mage-flow-multiref.png
```

## Python

```python
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    load_image_generation_model,
)

model = load_image_generation_model("microsoft/Mage-Flow")
result = generate_image(
    model,
    ImageGenerationRequest(
        prompt="A paper-cut city floating above a calm ocean",
        seed=7,
        steps=20,
        width=1024,
        height=1024,
        guidance=5.0,
    ),
)
result.save("outputs/mage-flow.png")
```

```python
from mlx_vlm.generate.edit_image import ImageEditRequest
from mlx_vlm.generate.image import generate_image, load_image_model

model = load_image_model("microsoft/Mage-Flow-Edit-Turbo", task="edit")
result = generate_image(
    model,
    ImageEditRequest(
        prompt="Turn the room into a watercolor illustration",
        image_paths=("input/room.png",),
        seed=11,
        steps=4,
        guidance=1.0,
    ),
    task="edit",
    max_size=1024,
)
result.save("outputs/mage-flow-edit.png")
```

Additional request options may be passed through `extra`:

- `negative_prompt` (default `" "`)
- `static_shift` (default `6.0`)
- `renormalization` (default `False`)
- `max_size` for edits
- `vl_cond_long_edge` for edit conditioning (default `384`)

The generic CLI defaults are four steps and guidance 1. Specify the recommended
settings above when using a Base or aligned checkpoint.
