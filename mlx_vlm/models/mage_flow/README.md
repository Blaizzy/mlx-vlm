# Mage-Flow

Mage-Flow is Microsoft's 4B native-resolution image generation and editing
family. The checkpoints are hosted by the
[`mage-flow-community`](https://huggingface.co/mage-flow-community)
organization. `mlx-vlm` loads the Hugging Face Diffusers-style checkpoints
directly and runs the Qwen3-VL conditioner, NR-MMDiT transformer, Mage-VAE, and
flow-matching sampler with MLX.

## Models

| Hugging Face model | Alias | Task | Recommended settings |
|---|---|---|---|
| `mage-flow-community/Mage-Flow-Base` | `mage-flow-base` | Generation | 30 steps, guidance 5 |
| `mage-flow-community/Mage-Flow` | `mage-flow` | Generation | 20 steps, guidance 5 |
| `mage-flow-community/Mage-Flow-Turbo` | `mage-flow-turbo` | Generation | 4 steps, guidance 1 |
| `mage-flow-community/Mage-Flow-Edit-Base` | `mage-flow-edit-base` | Editing | 30 steps, guidance 5 |
| `mage-flow-community/Mage-Flow-Edit` | `mage-flow-edit` | Editing | 30 steps, guidance 5 |
| `mage-flow-community/Mage-Flow-Edit-Turbo` | `mage-flow-edit-turbo` | Editing | 4 steps, guidance 1 |

All variants support native resolutions from 512 to 2048 pixels per side,
including aspect ratios up to 4:1. Width and height must be multiples of 16.

The first load downloads the checkpoint's transformer, VAE, text encoder,
tokenizer, and scheduler metadata. The text encoder is evicted after prompt
encoding by default to reduce peak resident memory before denoising.

## Quantization

Convert and quantize the diffusion-transformer blocks:

```sh
python -m mlx_vlm.models.mage_flow.convert \
  --hf-path mage-flow-community/Mage-Flow-Turbo \
  --mlx-path ./Mage-Flow-Turbo-MLX-4bit \
  --quantize \
  --q-mode affine \
  --q-bits 4 \
  --q-group-size 64
```

The converted directory can be passed to `--model` like the original
checkpoint. Supported modes are `affine`, `mxfp4`, `nvfp4`, and `mxfp8`.
The Qwen conditioner, transformer modulation/input/output projections, and VAE
remain in the selected floating-point dtype because quantizing these
quality-sensitive components causes severe image degradation.

## CLI

Generate with the aligned model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task generate \
  --model mage-flow-community/Mage-Flow \
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
  --model mage-flow-community/Mage-Flow-Edit \
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

model = load_image_generation_model("mage-flow-community/Mage-Flow")
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

model = load_image_model(
    "mage-flow-community/Mage-Flow-Edit-Turbo",
    task="edit",
)
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
