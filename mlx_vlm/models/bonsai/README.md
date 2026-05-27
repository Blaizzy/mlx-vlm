# Prism Bonsai

Prism Bonsai is an image generation model. In `mlx-vlm`, it uses the dedicated
image generation path rather than the text/VLM token generation path.

Capabilities:
- **Text-to-image generation** with the ternary MLX Bonsai model
- **CLI output** to PNG files
- **Python API output** as an evaluated `mx.array`
- **OpenAI-compatible API** through `/v1/images/generations`

## Models

| Model | Alias | Notes |
|-------|-------|-------|
| `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | `bonsai-ternary` | Ternary 2-bit MLX model |

The model is downloaded with `huggingface_hub.snapshot_download` when it is not
already available in the Hugging Face cache. You can also pass a local Bonsai
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
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A tiny glass bonsai tree on a moonlit desk" \
  --size 512x512 \
  --steps 4 \
  --seed 9909 \
  --output outputs/bonsai.png
```

### Equivalent `generate` command

```sh
python -m mlx_vlm generate \
  --output-modality image \
  --model bonsai-ternary \
  --prompt "A tiny glass bonsai tree on a moonlit desk" \
  --size 512x512 \
  --steps 4 \
  --seed 9909 \
  --output outputs/bonsai.png
```

If `--seed` is omitted, a random 32-bit seed is used. If `--output` is omitted,
the image is written to `outputs/image-{seed}.png`.

## Python

### Basic generation

```python
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    load_image_generation_model,
)

model = load_image_generation_model(
    "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
)

request = ImageGenerationRequest(
    prompt="A tiny glass bonsai tree on a moonlit desk",
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

result.save("outputs/bonsai.png")
```

### Prompt shorthand

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("bonsai-ternary")

result = generate_image(
    model,
    "A tiny glass bonsai tree on a moonlit desk",
    seed=9909,
    steps=4,
    width=512,
    height=512,
    output_path="outputs/bonsai.png",
)

print(result.path)
```

### Base64 PNG output

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("bonsai-ternary")
result = generate_image(
    model,
    "A tiny glass bonsai tree on a moonlit desk",
    seed=9909,
    steps=4,
    width=512,
    height=512,
)

b64_png = result.to_b64_json()
```

## API

```sh
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
    "prompt": "A tiny glass bonsai tree on a moonlit desk",
    "size": "512x512",
    "steps": 4,
    "seed": 9909,
    "response_format": "b64_json"
  }'
```

For local file output, set `"response_format": "path"` and optionally pass
`"output_path"` or `"output_dir"`.

## Notes

- Always pass an image generation model id or local snapshot path explicitly.
- The Bonsai tokenizer applies its own chat template; pass plain prompt text.
- The first supported variant is ternary. Binary/1-bit models are intentionally
  not exposed yet.
