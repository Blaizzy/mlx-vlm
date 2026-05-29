# FLUX.2 Klein

FLUX.2 Klein models are diffusion image models. In `mlx-vlm`, they use the
dedicated image generation and image editing paths rather than the text/VLM
token generation path.

Capabilities:
- **Text-to-image generation** with FLUX.2 Klein and FLUX.2 Klein Base models
- **Image editing** with FLUX.2 Klein 9B and the 9B KV reference-cache variant
- **CLI output** to PNG files
- **Python API output** as an evaluated `mx.array`
- **OpenAI-compatible API** through `/v1/images/generations` and `/v1/images/edits`

## Models

| Model | Alias | Generation | Editing | Notes |
|-------|-------|------------|---------|-------|
| `black-forest-labs/FLUX.2-klein-4B` | `flux2-klein-4b`, `flux2-klein`, `klein-4b` | Yes | No | Dense 4B text-to-image model |
| `black-forest-labs/FLUX.2-klein-9B` | `flux2-klein-9b`, `klein-9b` | Yes | Yes | Dense 9B text-to-image and image-edit model |
| `black-forest-labs/FLUX.2-klein-base-4B` | `flux2-klein-base-4b`, `flux2-base-4b`, `klein-base-4b` | Yes | No | Base 4B text-to-image model |
| `black-forest-labs/FLUX.2-klein-base-9B` | `flux2-klein-base-9b`, `flux2-base-9b`, `klein-base-9b` | Yes | No | Base 9B text-to-image model |
| `black-forest-labs/FLUX.2-klein-9b-kv` | `flux2-klein-9b-kv`, `klein-9b-kv` | Yes | Yes | 9B image-edit model with per-call reference-image KV caching |

Models are downloaded with `huggingface_hub.snapshot_download` when they are not
already available in the Hugging Face cache. You can also pass a local FLUX.2
snapshot path as `--model`.

Supported image sizes are 256 to 2048 pixels per side, and both dimensions must
be multiples of 16.

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Generate an image

```sh
python -m mlx_vlm generate_image \
  --model black-forest-labs/FLUX.2-klein-9B \
  --prompt "A cinematic portrait of a glass bonsai tree on a moonlit desk" \
  --size 512x512 \
  --steps 4 \
  --seed 9909 \
  --output outputs/flux2-klein.png
```

### Equivalent `generate` command

```sh
python -m mlx_vlm generate \
  --output-modality image \
  --model flux2-klein-9b \
  --prompt "A cinematic portrait of a glass bonsai tree on a moonlit desk" \
  --size 512x512 \
  --steps 4 \
  --seed 9909 \
  --output outputs/flux2-klein.png
```

### Edit an image

```sh
python -m mlx_vlm generate_image \
  --task edit \
  --model black-forest-labs/FLUX.2-klein-9B \
  --image input/person.png \
  --prompt "Add black sunglasses" \
  --size 512x512 \
  --steps 4 \
  --seed 123 \
  --output outputs/flux2-edit.png
```

### Edit an image with the KV-cache variant

```sh
python -m mlx_vlm generate_image \
  --task edit \
  --model black-forest-labs/FLUX.2-klein-9b-kv \
  --image input/person.png \
  --prompt "Add black sunglasses" \
  --size 512x512 \
  --steps 4 \
  --seed 123 \
  --output outputs/flux2-edit-kv.png
```

The standard 9B edit path concatenates reference tokens on every denoising step.
The `9b-kv` edit path extracts reference-image K/V on the first step and reuses
that cache for later steps.

If `--seed` is omitted, a random 32-bit seed is used. If `--output` is omitted,
generation writes to `outputs/image-{seed}.png` and editing writes to
`outputs/edit-{seed}.png`.

## Python

### Basic generation

```python
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    load_image_generation_model,
)

model = load_image_generation_model("black-forest-labs/FLUX.2-klein-9B")

request = ImageGenerationRequest(
    prompt="A cinematic portrait of a glass bonsai tree on a moonlit desk",
    seed=9909,
    steps=4,
    width=512,
    height=512,
    guidance=1.0,
)

result = generate_image(model, request)

# The primary Python output is an evaluated MLX array.
array = result.array
print(array.shape, array.dtype)

result.save("outputs/flux2-klein.png")
```

### Prompt shorthand

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("flux2-klein-9b")

result = generate_image(
    model,
    "A cinematic portrait of a glass bonsai tree on a moonlit desk",
    seed=9909,
    steps=4,
    width=512,
    height=512,
    output_path="outputs/flux2-klein.png",
)

print(result.path)
```

### Image editing

```python
from mlx_vlm.generate.image import generate_image, load_image_model

model = load_image_model("black-forest-labs/FLUX.2-klein-9b-kv", task="edit")

result = generate_image(
    model,
    "Add black sunglasses",
    task="edit",
    image_paths=("input/person.png",),
    seed=123,
    steps=4,
    width=512,
    height=512,
    guidance=1.0,
    output_path="outputs/flux2-edit-kv.png",
)
print(result.array.shape, result.path)
```

### Base64 PNG output

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("flux2-klein-4b")
result = generate_image(
    model,
    "A cinematic portrait of a glass bonsai tree on a moonlit desk",
    seed=9909,
    steps=4,
    width=512,
    height=512,
)

b64_png = result.to_b64_json()
```

## API

Start the server:

```sh
python -m mlx_vlm server
```

### Generate an image

```sh
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.2-klein-9B",
    "prompt": "A cinematic portrait of a glass bonsai tree on a moonlit desk",
    "size": "512x512",
    "steps": 4,
    "seed": 9909,
    "response_format": "b64_json"
  }'
```

### Edit an image

```sh
curl http://localhost:8080/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.2-klein-9b-kv",
    "image": "input/person.png",
    "prompt": "Add black sunglasses",
    "size": "512x512",
    "steps": 4,
    "seed": 123,
    "response_format": "b64_json"
  }'
```

For local file output, set `"response_format": "path"` and optionally pass
`"output_path"` or `"output_dir"`.

## Notes

- Image editing requires at least one local reference image path through
  `--image`, `ImageEditRequest.image_paths`, or the API `"image"` field.
- Multiple reference images are accepted for editing. The reference images are
  resized to the requested output size before VAE encoding.
