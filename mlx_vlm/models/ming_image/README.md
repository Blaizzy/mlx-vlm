# Ming-Image-0.1-Design (MLX)

MLX port of [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design),
a 6B text-to-image model for text-rich visual design (UI, posters, infographics)
with native RGBA / transparent-background output.

## Usage

```python
from mlx_vlm.models.ming_image import MingImagePipeline

pipe = MingImagePipeline.from_pretrained()  # downloads the checkpoint
image = pipe.generate_array(
    "a minimalist poster, bold word HELLO, blue background",
    seed=0, steps=12, width=1024, height=1024,
)  # -> [H, W, 4] uint8 RGBA
```

It is also dispatched by the shared image-generation entry point via the
`ming_image` model type (`inclusionAI/Ming-Image-0.1-Design` or a local path).

## Recommended settings

- Resolution **1024x1024** (fast) or **2048x2048**; width/height multiples of 16.
- Steps **12**, guidance **1.0** (the model is trained without classifier-free
  guidance; other values are rejected).
- Precision **bf16**.
- For a transparent background, prepend one of the model card's RGBA phrases
  (e.g. `RGBA, 4-channel, transparent background`); the 4-channel VAE decodes the
  alpha channel directly.

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

The caption embedding, context refiner, and RoPE tables carry no timestep
dependence, so they are computed once and reused across every denoising step
(`MingImageTransformer.prepare_conditioning` / `denoise`).
