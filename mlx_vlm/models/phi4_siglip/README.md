# Phi-4 Reasoning Vision (SigLIP2)

Microsoft's Phi-4 multimodal reasoning model combining a Phi-3 language backbone with a SigLIP2 NaFlex vision encoder. Supports variable-resolution image understanding with strong reasoning capabilities.

## Model

| | |
|---|---|
| **Model ID** | `microsoft/Phi-4-reasoning-vision-15B` |
| **Architecture** | Phi-3 (language) + SigLIP2 NaFlex (vision) + MLP 2x GELU projector |
| **Parameters** | ~15B |
| **Vision Encoder** | SigLIP2 with NaFlex (variable resolution, 256-3600 patches) |
| **Tasks** | Visual question answering, image reasoning, image description |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model microsoft/Phi-4-reasoning-vision-15B \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

With custom generation parameters:

```bash
python -m mlx_vlm.generate \
    --model microsoft/Phi-4-reasoning-vision-15B \
    --image path/to/image.jpg \
    --prompt "What objects are in this image?" \
    --max-tokens 512 \
    --temp 0.0
```

## Python Usage

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image_processor

model, processor = load("microsoft/Phi-4-reasoning-vision-15B")

image = "path/to/image.jpg"
prompt = "Describe this image."

formatted = apply_chat_template(processor, model.config, prompt, num_images=1)

output = generate(
    model,
    processor,
    formatted,
    [image],
    max_tokens=512,
    temperature=0.0,
)
print(output)
```

## Architecture

- **Vision**: SigLIP2 with NaFlex dynamic patching. Images are processed into a variable number of patches (256-3600) based on resolution, enabling efficient handling of different aspect ratios.
- **Projector**: Two-layer MLP with GELU activation that maps vision features (1152-dim) to the language model's hidden space (5120-dim).
- **Language**: Phi-3 decoder with 40 layers, grouped-query attention (40 heads, 10 KV heads), fused QKV/gate-up projections, and RoPE.

## Notes

- The model uses `<image>` as the image placeholder token with `IMAGE_TOKEN_INDEX = -200`.
- NaFlex produces variable-length image feature sequences per image, so the effective context length depends on the input image resolution.
- The vision encoder uses bicubic interpolation for dynamic positional embedding resizing.
