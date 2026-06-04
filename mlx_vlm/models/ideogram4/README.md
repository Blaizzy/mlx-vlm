# Ideogram 4

Ideogram 4 is a non-commercial text-to-image model.

Capabilities:
- **Text-to-image generation** with the Ideogram 4 model
- **Structured JSON captions** for layout, style, palette, and typography control
- **OpenAI-compatible API** through `/v1/images/generations`

## Supported Models

| Model | Quantization | Generation | Notes |
|-------|--------------|------------|-------|
| `ideogram-ai/ideogram-4-fp8` | FP8 weights, dequantized to BF16 at load time | Yes | Gated Hugging Face repo, Ideogram 4 Non-Commercial license |

Supported image sizes are 256 to 2048 pixels per side, both dimensions must be
multiples of 16, and aspect ratio is limited to 6:1.

## Install

```sh
pip install -U mlx-vlm
```

Accept the Hugging Face gate before first use and make a token available through
`HF_TOKEN` or `huggingface_hub` login.

## CLI

### Generate an image

```sh
python -m mlx_vlm generate_image \
  --model ideogram-ai/ideogram-4-fp8 \
  --prompt '{"high_level_description":"A clean square poster with the words MLX VLM.","compositional_deconstruction":{"background":"A white poster background.","elements":[{"type":"text","text":"MLX VLM","desc":"Large crisp black title text centered on the poster."}]}}' \
  --size 1024x1024 \
  --seed 42 \
  --output outputs/ideogram4.png \
  --gen-kwargs '{"sampler_preset":"V4_DEFAULT_20"}'
```

### Equivalent `generate` command

```sh
python -m mlx_vlm generate \
  --output-modality image \
  --model ideogram-ai/ideogram-4-fp8 \
  --prompt '{"high_level_description":"A clean square poster with the words MLX VLM.","compositional_deconstruction":{"background":"A white poster background.","elements":[{"type":"text","text":"MLX VLM","desc":"Large crisp black title text centered on the poster."}]}}' \
  --size 1024x1024 \
  --seed 42 \
  --output outputs/ideogram4.png \
  --gen-kwargs '{"sampler_preset":"V4_DEFAULT_20"}'
```

Use `V4_QUALITY_48` for the highest-quality preset, `V4_DEFAULT_20` for the
default local preset, or `V4_TURBO_12` for a smaller smoke test. If `--seed` is
omitted, a random 32-bit seed is used. If `--output` is omitted, the image is
written to `outputs/image-{seed}.png`.

## Python

### Basic generation

```python
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    load_image_generation_model,
)

model = load_image_generation_model("ideogram-ai/ideogram-4-fp8")

request = ImageGenerationRequest(
    prompt='{"high_level_description":"A red square icon.","compositional_deconstruction":{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}',
    seed=0,
    width=1024,
    height=1024,
    extra={"sampler_preset": "V4_DEFAULT_20"},
)

result = generate_image(model, request)

# The primary Python output is an evaluated MLX array.
array = result.array
print(array.shape, array.dtype)

result.save("outputs/ideogram4.png")
```

### Prompt shorthand

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("ideogram-ai/ideogram-4-fp8")

result = generate_image(
    model,
    '{"high_level_description":"A red square icon.","compositional_deconstruction":{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}',
    seed=0,
    width=1024,
    height=1024,
    output_path="outputs/ideogram4.png",
    sampler_preset="V4_DEFAULT_20",
)

print(result.path)
```

### Base64 PNG output

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("ideogram-ai/ideogram-4-fp8")
result = generate_image(
    model,
    '{"high_level_description":"A red square icon.","compositional_deconstruction":{"background":"A white background.","elements":[{"type":"obj","desc":"A centered red square."}]}}',
    seed=0,
    width=1024,
    height=1024,
    sampler_preset="V4_DEFAULT_20",
)

b64_png = result.to_b64_json()
```

## API

Start the server:

```sh
python -m mlx_vlm server
```

Generate an image:

```sh
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ideogram-ai/ideogram-4-fp8",
    "prompt": "{\"high_level_description\":\"A red square icon.\",\"compositional_deconstruction\":{\"background\":\"A white background.\",\"elements\":[{\"type\":\"obj\",\"desc\":\"A centered red square.\"}]}}",
    "size": "1024x1024",
    "seed": 0,
    "response_format": "b64_json"
  }'
```

For local file output, set `"response_format": "path"` and optionally pass
`"output_path"` or `"output_dir"`.

## Notes

- The runtime does not call external prompt-expansion or safety moderation
  services.
- Plain prompts are accepted, but structured JSON captions usually give better
  quality and control.
  