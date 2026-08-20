# ERNIE-Image

ERNIE-Image is Baidu's text-to-image model family. `mlx-vlm` runs the
Ministral-3 text encoder, ERNIE-Image diffusion transformer, FLUX.2 VAE, and
flow-matching scheduler with MLX. It also supports optional prompt enhancement.

## Models

| Hugging Face model | Variant | Generation | Image-to-image | Recommended settings |
|---|---|---|---|---|
| `baidu/ERNIE-Image-Turbo` | Turbo | Yes | Yes | 8 steps, guidance 1 |
| `baidu/ERNIE-Image` | Base | Yes | Yes | 50 steps, guidance 4 |

The base model uses classifier-free guidance. Turbo disables classifier-free
guidance at its recommended guidance value. Width and height must be positive
multiples of 16.

Image-to-image uses generic SDEdit-style latent transformation rather than
native instruction-edit conditioning. It accepts exactly one source image.

## Convert

Convert the Turbo checkpoint to BF16:

```sh
python -m mlx_vlm.models.ernie_image.convert \
  --hf-path baidu/ERNIE-Image-Turbo \
  --mlx-path ./ERNIE-Image-Turbo-MLX
```

Convert and quantize compatible model layers:

```sh
python -m mlx_vlm.models.ernie_image.convert \
  --hf-path baidu/ERNIE-Image-Turbo \
  --mlx-path ./ERNIE-Image-Turbo-MLX-MXFP8 \
  --quantize \
  --q-mode mxfp8
```

Supported quantization modes are `affine`, `mxfp4`, `nvfp4`, and `mxfp8`.
Mode-specific defaults are used when `--q-bits` and `--q-group-size` are
omitted.

## CLI

Generate an image with the Turbo model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model ./ERNIE-Image-Turbo-MLX \
  --prompt "A red panda serving tea in a moonlit bamboo forest" \
  --size 1024x1024 \
  --steps 8 \
  --guidance 1 \
  --seed 42 \
  --output outputs/ernie-image-turbo.png
```

Generate an image with the Base model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model ./ERNIE-Image-MLX \
  --prompt "A cinematic photograph of a fox walking through fresh snow" \
  --size 1024x1024 \
  --steps 50 \
  --guidance 4 \
  --seed 42 \
  --output outputs/ernie-image.png \
  --gen-kwargs '{"negative_prompt":"blurry, low quality"}'
```

Transform one source image:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task edit \
  --model ./ERNIE-Image-Turbo-MLX \
  --image input/fox.png \
  --prompt "Give the fox a red Santa hat" \
  --size 1024x1024 \
  --steps 8 \
  --guidance 1 \
  --seed 42 \
  --output outputs/ernie-image-edit.png \
  --gen-kwargs '{"image_strength":0.6}'
```

`image_strength` must be greater than 0 and no more than 1. The optional prompt
enhancer is used automatically when its weights are present.

## Python

Generate an image:

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("./ERNIE-Image-Turbo-MLX")
result = generate_image(
    model,
    "A red panda serving tea in a moonlit bamboo forest",
    seed=42,
    steps=8,
    width=1024,
    height=1024,
    guidance=1.0,
    output_path="outputs/ernie-image-turbo.png",
)

print(result.array.shape, result.path)
```

Transform an image:

```python
from mlx_vlm.generate.image import generate_image, load_image_model

model = load_image_model("./ERNIE-Image-Turbo-MLX", task="edit")
result = generate_image(
    model,
    "Give the fox a red Santa hat",
    task="edit",
    image_paths=("input/fox.png",),
    seed=42,
    steps=8,
    guidance=1.0,
    image_strength=0.6,
    output_path="outputs/ernie-image-edit.png",
)

print(result.array.shape, result.path)
```

The primary Python output is an evaluated `mx.array`. The result can also be
saved later with `result.save(path)` or encoded as a base64 PNG with
`result.to_b64_json()`.
