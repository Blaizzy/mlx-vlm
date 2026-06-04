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

## Prompting

Ideogram 4 was trained on structured JSON captions. Existing JSON object prompts
are passed through unchanged, while ordinary text prompts are wrapped into a
minimal JSON caption by default. Set `auto_json_caption=False` to pass raw plain
text through without wrapping.

For richer captions, `prompt_expansion_model` can use any compatible local
mlx-vlm text/VLM model to expand an ordinary prompt into structured Ideogram 4
JSON before image generation. This is a local mlx-vlm prompt expansion path,
not Ideogram's hosted production Magic Prompt.

If the expansion model returns a valid structured JSON caption, it is used as
the Ideogram prompt and exposed as `revised_prompt`. If its output is not a
valid structured caption, mlx-vlm falls back to the minimal JSON caption wrapper
unless `auto_json_caption=False` is set. Model loading and inference errors are
raised so configuration problems are not hidden.

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

### Expand a plain prompt locally

```sh
python -m mlx_vlm generate_image \
  --model ideogram-ai/ideogram-4-fp8 \
  --prompt "A cinematic photo of a glass teapot on a rainy London cafe table" \
  --prompt-expansion-model mlx-community/gemma-4-12B-it-4bit \
  --size 1024x1024 \
  --seed 42 \
  --output outputs/ideogram4-expanded.png \
  --gen-kwargs '{"sampler_preset":"V4_DEFAULT_20"}'
```

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

### Expand a plain prompt locally

```python
request = ImageGenerationRequest(
    prompt="A cinematic photo of a glass teapot on a rainy London cafe table",
    seed=0,
    width=1024,
    height=1024,
    extra={
        "sampler_preset": "V4_DEFAULT_20",
        "prompt_expansion_model": "mlx-community/gemma-4-12B-it-4bit",
    },
)

result = generate_image(model, request)
print(result.metadata["revised_prompt"])
```

To pass raw plain text without JSON wrapping:

```python
request = ImageGenerationRequest(
    prompt="A cinematic photo of a glass teapot on a rainy London cafe table",
    extra={"auto_json_caption": False},
)
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

The API also accepts the optional Ideogram 4 fields `auto_json_caption` and
`prompt_expansion_model`. When a prompt is wrapped or expanded, the response
includes the resulting JSON caption in `revised_prompt`.

## Notes

- The runtime does not call external hosted prompt-expansion or safety moderation
  services.
- Plain prompts are accepted, but structured JSON captions usually give better
  quality and control.
