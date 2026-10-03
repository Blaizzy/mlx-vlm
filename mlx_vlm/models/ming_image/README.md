# Ming-Image-0.1-Design (MLX)

MLX port of [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design),
a 6B text-to-image model for text-rich visual design (UI, posters, infographics)
with native RGBA / transparent-background output.

## Supported repo IDs

| Hugging Face model | Precision | Notes |
|---|---|---|
| `inclusionAI/Ming-Image-0.1-Design` | bf16 | Original Diffusers checkpoint; loads directly (sanitized on the fly). |
| `nativ-community/Ming-Image-0.1-Design-MLX-4bit` | 4-bit | Smallest download, lowest peak memory. |
| `nativ-community/Ming-Image-0.1-Design-MLX-8bit` | 8-bit | Higher fidelity than 4-bit, still quantized. |

The original checkpoint loads directly; you can also pass a local checkpoint or
MLX-converted model directory.

## Recommended settings

- Resolution **1024x1024** (fast) or **2048x2048**; width/height multiples of 16.
- Steps **12**, guidance **1.0** (the model is trained without classifier-free
  guidance; other values are rejected).
- For a transparent background, prepend one of the model card's RGBA phrases
  (e.g. `RGBA, 4-channel, transparent background`); the 4-channel VAE decodes the
  alpha channel directly.

## CLI

```sh
mlx_vlm.generate \
  --output-modality image \
  --model nativ-community/Ming-Image-0.1-Design-MLX-4bit \
  --prompt "a minimalist poster, bold word HELLO, blue background" \
  --size 1024x1024 \
  --steps 12 \
  --guidance 1 \
  --seed 0 \
  --output outputs/ming-image.png
```

## Python

```python
from mlx_vlm.generate.image import generate_image, load_image_generation_model

model = load_image_generation_model("nativ-community/Ming-Image-0.1-Design-MLX-4bit")
result = generate_image(
    model,
    "a minimalist poster, bold word HELLO, blue background",
    seed=0,
    steps=12,
    width=1024,
    height=1024,
    guidance=1.0,
    output_path="outputs/ming-image.png",
)

print(result.array.shape, result.path)
```

## Convert

The bf16 checkpoint runs from the original repo, so conversion is only needed to
reproduce the quantized variants above. Point `--model` at a local copy of the
original checkpoint:

```sh
python -m mlx_vlm.models.ming_image.convert \
  --model /path/to/Ming-Image-0.1-Design \
  --output ./Ming-Image-0.1-Design-MLX-4bit \
  --bits 4 --group-size 64
```

## Architecture

The conditioning is produced once per prompt:

- a **BailingMoeV2** MoE LLM (`mllm`) encodes the templated prompt plus 256
  learnable query tokens; its MoE MultiRouter routes the query tokens through
  `image_gate` and the text tokens through `gate`;
- a bidirectional **Qwen2** connector turns the query-token states into 256
  caption features (`cap_feats`, 2560-dim), and the concatenated hidden states of
  layers `[5, 12, 20]` become a direct-VLM stream (`cap_feats_2`, 3840-dim).

A Lumina/NextDiT **diffusion transformer** (`transformer`) then denoises the
latent with joint image+caption self-attention over ~12 flow-match steps, and an
`AutoencoderKLQwenImage` decodes the latent to a 4-channel RGBA image.
