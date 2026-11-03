# MOSS-Image

MLX text-to-image inference for both MOSS-Image checkpoints:

| Model | Description |
| --- | --- |
| [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | Base text-to-image model |
| [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | Supervised fine-tuned checkpoint |

Load the original Hugging Face checkpoint directly. The implementation uses the
MOSS diffusion transformer, Qwen3.5's final text hidden states, and FLUX.2 VAE.
Runtime inference requires no PyTorch or Diffusers.

```sh
mlx_vlm.generate --output-modality image --model OpenMOSS-Team/MOSS-Image-Base \
  --prompt "A red ceramic teapot on a wooden table, soft morning light" \
  --size 1024x1024 --seed 42 --output moss.png
```

```python
from mlx_vlm.generate import generate_image, load_image_generation_model

model = load_image_generation_model("OpenMOSS-Team/MOSS-Image-SFT")
result = generate_image(
    model,
    "A red ceramic teapot on a wooden table, soft morning light",
    width=1024,
    height=1024,
    seed=42,
)
result.save("moss.png")
```

Defaults are 50 sampling steps, guidance 3.0, scheduler shift 5.0, and an empty
negative prompt for both Base and SFT. Set dimensions explicitly; both must be
positive multiples of 16. The generic image API defaults to 512×512, while
upstream recommends 1024×1024.
The text encoder is released before loading the diffusion transformer by default.
The original weights occupy approximately 27 GB on disk.

The Python API accepts `negative_prompt`, `max_sequence_length` (up to 4096),
`num_images`, `uncond_skip_layers`, `slg_layers`, `slg_scale`, `slg_start`, and
`slg_stop` in `ImageGenerationRequest.extra` or as `generate_image` keyword
arguments. The upstream default skips block 22 in the negative CFG branch.
Use `uncond_skip_layers=[]` to disable this. Optional positive skip-layer guidance
uses `slg_layers` and `slg_scale`; its active interval is `[slg_start, slg_stop)`
in denoising order. Guidance 1 disables the negative branch. Generation supports
plain captions and JSON layout prompts; pass layout JSON as the prompt string.

The MOSS-Image weights and upstream reference implementation are licensed under
[Apache 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE).
