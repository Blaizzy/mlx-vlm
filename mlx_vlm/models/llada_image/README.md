# LLaDA-Image

Native MLX generation and single-image editing for the base and Turbo models,
including their official BF16 and FP8 checkpoints:

- `inclusionAI/LLaDA-Image`
- `inclusionAI/LLaDA-Image-Turbo`
- `inclusionAI/LLaDA-Image-FP8`
- `inclusionAI/LLaDA-Image-Turbo-FP8`

Checkpoints load directly without conversion or executing custom Python code.

```sh
mlx_vlm.generate \
  --output-modality image \
  --model inclusionAI/LLaDA-Image-Turbo \
  --prompt "A red panda serving tea in a moonlit bamboo forest" \
  --size 1024x1024 \
  --seed 42 \
  --output llada-image.png
```

Turbo defaults to four steps, guidance 1, and its stochastic uniform schedule.
The base model defaults to 50 steps, guidance 5, and its nonlinear Euler schedule.
Width and height must be multiples of 16. `negative_prompt` applies when guidance
is greater than 1. Pass `--gen-kwargs '{"stochastic_sampling":false}'` to disable
stochastic sampling without changing the checkpoint's sigma schedule.

## VQ-guided generation

The default `text` mode uses prompt embeddings. Set `generation_mode` to `vq` to
also generate discrete image tokens with the LLaDA text backbone and condition
the denoiser on their SigVQ features. This extra sampling stage is slower.

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("inclusionAI/LLaDA-Image-Turbo")
result = generate_image(
    model,
    "A red ceramic teapot on a wooden table",
    width=1024,
    height=1024,
    seed=42,
    generation_mode="vq",
)
result.save("llada-vq.png")
```

For the CLI, pass `--gen-kwargs '{"generation_mode":"vq"}'`.

## Native editing

Editing conditions the denoiser on both the source image's VAE latents and SigVQ
features. Exactly one source image is supported. Explicit dimensions must be
multiples of 32; omitted dimensions preserve the source aspect ratio, rounded
to that grid, with a default maximum area of 1024×1024.

```python
from mlx_vlm.generate.edit_image import edit_image, load_image_edit_model

model = load_image_edit_model("inclusionAI/LLaDA-Image-Turbo")
result = edit_image(
    model,
    "Make the teapot blue, preserving the rest of the image",
    image_paths=["llada-vq.png"],
    seed=42,
)
result.save("llada-edit.png")
```

The CLI also supports `--task edit --image source.png`.
