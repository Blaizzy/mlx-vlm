# Z-Image

Z-Image is Tongyi-MAI's text-to-image model family. `mlx-vlm` converts the
original Hugging Face Diffusers checkpoints to MLX format and runs the Qwen3
text encoder, Z-Image diffusion transformer, VAE, and flow-matching scheduler
with MLX.

## Models

| Hugging Face model | Variant | Generation | Image-to-image | Recommended settings |
|---|---|---|---|---|
| `Tongyi-MAI/Z-Image-Turbo` | Turbo | Yes | Yes | 9 steps, guidance 0 |
| `Tongyi-MAI/Z-Image` | Base | Yes | Yes | 50 steps, guidance 4 |

The model must be converted before use. Generation and image-to-image commands
accept the resulting local MLX directory as `--model`. Width and height must be
positive multiples of 16.

## Convert

Convert the Turbo checkpoint to BF16:

```sh
python -m mlx_vlm.models.z_image.convert \
  --hf-path Tongyi-MAI/Z-Image-Turbo \
  --mlx-path ./Z-Image-Turbo-MLX
```

Convert and quantize the transformer and text encoder:

```sh
python -m mlx_vlm.models.z_image.convert \
  --hf-path Tongyi-MAI/Z-Image-Turbo \
  --mlx-path ./Z-Image-Turbo-MLX-4bit \
  --quantize \
  --q-mode affine \
  --q-bits 4 \
  --q-group-size 64
```

The VAE remains in the selected floating-point dtype by default. Add
`--quantize-vae` to quantize compatible VAE layers as well.

## CLI

Generate an image with the Turbo model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model ./Z-Image-Turbo-MLX \
  --prompt "A red panda serving tea in a moonlit bamboo forest" \
  --size 1024x1024 \
  --steps 9 \
  --guidance 0 \
  --seed 42 \
  --output outputs/z-image-turbo.png
```

Generate an image with the Base model:

```sh
mlx_vlm.generate \
  --output-modality image \
  --model ./Z-Image-MLX \
  --prompt "A cinematic photograph of a fox walking through fresh snow" \
  --size 1024x1024 \
  --steps 50 \
  --guidance 4 \
  --seed 42 \
  --output outputs/z-image.png \
  --gen-kwargs '{"negative_prompt":"blurry, low quality","cfg_truncation":1.0}'
```

Edit one image with the image-to-image path:

```sh
mlx_vlm.generate \
  --output-modality image \
  --task edit \
  --model ./Z-Image-Turbo-MLX \
  --image input/fox.png \
  --prompt "Give the fox a red Santa hat" \
  --size 1024x1024 \
  --steps 8 \
  --guidance 0 \
  --seed 42 \
  --output outputs/z-image-edit.png \
  --gen-kwargs '{"strength":0.6}'
```

Turbo does not support classifier-free guidance; use a guidance value from 0
to 1 to keep it disabled. Base supports classifier-free guidance and accepts
`negative_prompt` and `cfg_truncation` through `--gen-kwargs`. Image-to-image
accepts exactly one source image, and `strength` must be greater than 0 and no
more than 1.

## Python

Generate an image:

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("./Z-Image-Turbo-MLX")
result = generate_image(
    model,
    "A red panda serving tea in a moonlit bamboo forest",
    seed=42,
    steps=9,
    width=1024,
    height=1024,
    guidance=0.0,
    output_path="outputs/z-image-turbo.png",
)

print(result.array.shape, result.path)
```

Edit an image:

```python
from mlx_vlm.generate.image import generate_image, load_image_model

model = load_image_model("./Z-Image-Turbo-MLX", task="edit")
result = generate_image(
    model,
    "Give the fox a red Santa hat",
    task="edit",
    image_paths=("input/fox.png",),
    seed=42,
    steps=8,
    guidance=0.0,
    strength=0.6,
    output_path="outputs/z-image-edit.png",
)

print(result.array.shape, result.path)
```

The primary Python output is an evaluated `mx.array`. The result can also be
saved later with `result.save(path)` or encoded as a base64 PNG with
`result.to_b64_json()`.
