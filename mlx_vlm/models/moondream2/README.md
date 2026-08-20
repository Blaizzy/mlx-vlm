# Moondream2

Moondream2 is a compact vision-language model for visual question answering, image captioning, and visual reasoning. It uses a ViT-style vision encoder with a Transformer decoder for efficient multimodal understanding.

## Supported Models

| Model ID | Notes |
|---|---|
| `vikhyatk/moondream2` | Official Moondream2 repository |

## Model

| | |
|---|---|
| **Model ID** | `vikhyatk/moondream2` |
| **Architecture** | ViT-style vision encoder + Transformer decoder |
| **Vision Encoder** | 27 layers, 1152 dim, 16 heads, patch size 14, crop size 378 |
| **Language Model** | 24 layers, 2048 dim, 32 heads |
| **Tasks** | Visual question answering, image description, visual reasoning |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model vikhyatk/moondream2 \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200
```

With custom parameters:

```bash
python -m mlx_vlm.generate \
    --model vikhyatk/moondream2 \
    --image path/to/image.jpg \
    --prompt "How many objects are in this image?" \
    --max-tokens 100 \
    --temp 0.0
```

## Python Usage

```python
from mlx_vlm import load, generate

model, processor = load("vikhyatk/moondream2")

output = generate(
    model,
    processor,
    "Describe this image",
    ["path/to/image.jpg"],
    max_tokens=200,
)
print(output)
```

## Architecture

- **Vision**: ViT-style encoder with multi-crop support. Images are patchified into 14x14 patches on 378x378 crops, then projected to the language hidden size.
- **Language**: Transformer decoder with 24 layers, 2048 hidden size, and 32 attention heads.
- **Prompt format**: Registered as a prompt-only model in MLX-VLM.
- **Weight loading**: Includes a sanitize mapping for the official Moondream2 checkpoint key layout.

## Notes

- Region-model weights are ignored by the current text-generation path.
- Multi-crop image processing follows the Moondream2 crop reconstruction flow.
