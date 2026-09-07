# DINOv2 (MLX)

Channel-last MLX port of the [DINOv2](https://github.com/facebookresearch/dinov2)
vision transformer. The package holds:

- `Model` — the standalone image encoder. Loads the Hugging Face
  `facebook/dinov2-{small,base,large,giant}` and
  `facebook/dinov2-with-registers-{small,base,large,giant}` checkpoints
  directly (`dinov2_with_registers` configs are remapped to this package).
  Register tokens, patch masks (`bool_masked_pos`), `forward_features` and
  `get_intermediate_layers` are supported.
- `DINOv2` / `DINOv2Encoder` — the shared backbone used by dense-prediction
  models (see below).

The training-time DINO heads and stochastic depth (drop path) are not ported.

## Example

```python
import mlx.core as mx
from mlx_vlm import load

model, processor = load("facebook/dinov2-with-registers-base")

# processor returns channel-first pixel values; the model takes (B, H, W, C)
pixel_values = processor(images=[image], return_tensors="np")["pixel_values"]
out = model(mx.array(pixel_values).transpose(0, 2, 3, 1))
out["last_hidden_state"]      # (B, 1 + registers + patches, D), normed
out["pooler_output"]          # (B, D) — normed cls token
out["hidden_patch_tokens"]    # (B, patches, D)
```

Positional-embedding interpolation follows the Hugging Face reference
(size-based, antialiased when the config sets `interpolate_antialias`). Set
`interpolate_offset: 0.1` in the config for the original repo's scale-factor
behavior (used by the MoGe-3 and Video Depth Anything backbones below).

## Used by

| Model | Backbones |
| --- | --- |
| `video_depth_anything` | ViT-S/14, ViT-B/14, ViT-L/14 |
| `moge3` | ViT-L/14, ViT-g/14 |
