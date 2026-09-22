# Qwen-Image-2.1

Qwen-Image-2.1 supports text-to-image generation and instruction-based editing
with one or more reference images. `mlx-vlm` runs its Qwen3-VL conditioner,
diffusion transformer, VAE, and flow-matching sampler with MLX. Editing supports
RGBA images and preserves transparency in PNG output.

## Supported repo IDs

| Hugging Face model | Alias | Generation | Editing |
|---|---|---|---|
| `Qwen/Qwen-Image-2.1` | `qwen-image-2.1` | Yes | Yes |

The original Diffusers checkpoint loads directly. You can also pass a local
checkpoint or MLX-converted model directory.

## CLI

Generate an image:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A red ceramic teapot on a wooden table in soft window light" \
  --size 1024x1024 \
  --steps 30 \
  --guidance 1 \
  --seed 7 \
  --output outputs/qwen-image.png
```

Edit an image:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task edit \
  --model Qwen/Qwen-Image-2.1 \
  --image input/teapot.png \
  --prompt "Change the teapot to cobalt blue. Keep its shape and background unchanged." \
  --size 1024x1024 \
  --steps 40 \
  --guidance 1 \
  --seed 7 \
  --output outputs/qwen-image-edit.png
```

`--image` accepts multiple paths for editing with multiple references.

## Python

Generate an image:

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("Qwen/Qwen-Image-2.1")
result = generate_image(
    model,
    "A red ceramic teapot on a wooden table in soft window light",
    width=1024,
    height=1024,
    steps=30,
    guidance=1.0,
    seed=7,
    output_path="outputs/qwen-image.png",
)
```

Edit an image:

```python
from mlx_vlm.generate.edit_image import edit_image, load_image_edit_model

model = load_image_edit_model("Qwen/Qwen-Image-2.1")
result = edit_image(
    model,
    "Change the teapot to cobalt blue. Keep its shape and background unchanged.",
    image_paths=["input/teapot.png"],
    steps=40,
    guidance=1.0,
    seed=7,
    output_path="outputs/qwen-image-edit.png",
)
```

Add paths to `image_paths` for multiple references. Editing defaults to roughly
1024×1024 pixels using the last reference's aspect ratio. Set `width` and `height`
to override the output dimensions, or `output_resolution=512` to reduce both the
reference area and default output area. `result.array` contains the evaluated MLX
array; `result.save(path)` saves it as a PNG.
