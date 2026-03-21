# MolmoPoint

MolmoPoint is a vision-language model with pixel-precise pointing capabilities. Given an image and a text prompt like "Point to the cats", it generates exact (x, y) coordinates for the requested objects.

## Model

| | |
|---|---|
| **Model ID** | `allenai/MolmoPoint-8B` |
| **Architecture** | SigLIP ViT (vision) + Attention Pooling Connector + Qwen2-style Decoder (language) + PointPredictor |
| **Parameters** | ~8B |
| **Vision Encoder** | 27 layers (truncated to 25), 1152 dim, 16 heads, head dim 72, patch size 14, input 378x378 |
| **Language Model** | 36 layers, 4096 dim, 32 heads, 8 KV heads, head dim 128, SwiGLU MLP |
| **Connector** | Cross-attention pooling (no output projection) + gated SiLU MLP (1152 -> 12288 -> 4096) |
| **Point Predictor** | 3-stage: patch selection (RoPE keys) -> subpatch selection -> 3x3 location grid |
| **Tasks** | Visual pointing/grounding, image description, visual question answering |


## How Pointing Works

MolmoPoint extends the standard vocabulary with **patch**, **subpatch**, and **location** tokens:

1. **Patch selection** -- The model selects which image patch contains the target object. Patch keys are built from image token hidden states with 1D rotary embeddings. A learnable "no more points" class signals the end of pointing.
2. **Subpatch selection** -- Within the selected patch, the model picks a specific ViT sub-patch using raw ViT features (before pooling).
3. **Location refinement** -- A 3x3 grid refines the point within the sub-patch for sub-patch-level precision.

Each point is encoded as a triple `<POINT_patch> <POINT_subpatch> <POINT_location> object_id`.

## CLI Usage

### Image description

```bash
python -m mlx_vlm.generate \
    --model allenai/MolmoPoint-8B \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200
```

### Point to objects

```bash
python -m mlx_vlm.generate \
    --model allenai/MolmoPoint-8B \
    --image path/to/image.jpg \
    --prompt "Point to the cats" \
    --max-tokens 50
```

## Python Usage

### Basic generation

```python
from mlx_vlm import load, generate

model, processor = load("allenai/MolmoPoint-8B")

output = generate(
    model,
    processor,
    "Describe this image",
    ["path/to/image.jpg"],
    max_tokens=200,
)
print(output)
```

### Point extraction and visualization

```python
import mlx.core as mx
from mlx_vlm.utils import load, prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.generate import generate_step
from mlx_vlm.models.molmo_point.point_utils import (
    extract_points_from_text,
    draw_points_on_image,
)

model, processor = load("allenai/MolmoPoint-8B")
mx.eval(model.parameters())

image_path = "path/to/image.jpg"
prompt = apply_chat_template(
    processor, model.config, "Point to the cats", num_images=1
)
inputs = prepare_inputs(processor, images=image_path, prompts=prompt)

input_ids = inputs["input_ids"]
pixel_values = inputs.get("pixel_values", None)
mask = inputs.get("attention_mask", None)
kwargs = {
    k: v
    for k, v in inputs.items()
    if k not in ["input_ids", "pixel_values", "attention_mask"]
}

# Generate
tokens = []
for n, (token, _) in enumerate(
    generate_step(input_ids, model, pixel_values, mask, max_tokens=50, **kwargs)
):
    tokens.append(token)
    if n >= 49:
        break

output_text = processor.tokenizer.decode(tokens)
print(output_text)

# Extract points from the generated text
if hasattr(processor, "_pointing_metadata") and processor._pointing_metadata:
    points = extract_points_from_text(
        output_text,
        processor._pointing_metadata,
        no_more_points_class=model.config.no_more_points_class,
        patch_location=model.config.patch_location,
    )
    for obj_id, img_num, x, y in points:
        print(f"Object {obj_id}: ({x:.1f}, {y:.1f})")

    # Save annotated image
    draw_points_on_image(image_path, points, "output_pointed.jpg")
```
## Folder Structure

```
mlx_vlm/models/molmo_point/
    __init__.py                  # Module exports
    config.py                    # ModelConfig, TextConfig, VisionConfig, AdapterConfig
    vision.py                    # SigLIP ViT encoder (Linear patch embedding, LayerNorm)
    language.py                  # Qwen2-style LLM decoder (RMSNorm, GQA, QK-norm, SwiGLU)
    molmo_point.py               # Main Model: connector, point predictor, extended LM head,
                                 #   logit processor, image cache, generation logic
    image_processing.py          # Image processor (overlapping crops, tiling, pooling; no torch)
    processing_molmo_point.py    # Tokenizer + image token insertion
    point_utils.py               # Point extraction from <POINT_X> tokens + visualization
```

## Architecture Details

- **Image processing**: Images are decomposed into overlapping crops (up to 8, overlap margins [4, 4]) at 378x378, plus a global resized view. Each crop is patchified into 27x27 = 729 patches of 14x14x3 = 588 pixels. Patches from layers 18 and 24 of the ViT are concatenated (2304-dim features).
- **Connector**: Attention-pooled features (mean query, cross-attention without output projection) are projected through a gated SiLU MLP to the LLM dimension (4096).
- **LLM**: Qwen2-style decoder with fused QKV projection, per-head RMSNorm on Q/K (Qwen3-style), and SwiGLU MLP. RoPE with theta=1M.
- **Extended vocabulary**: The standard vocab (151,936 + 128 additional) is extended at runtime with patch, subpatch, and location tokens. A logit processor enforces valid generation order (patch -> subpatch -> location).

## Notes

- The custom image processor uses PIL and numpy only (no torch/torchvision dependency).
- Point extraction requires the `_pointing_metadata` stored on the processor after calling `prepare_inputs` with images.
- Peak memory is approximately 39 GB for the full bf16 model.
- Generation speed is ~6 tokens/sec on Apple Silicon (M-series).
