# MoGe-3 (MLX)

Port of [MoGe-3](https://github.com/microsoft/MoGe): fine-detail monocular
geometry estimation with self-guided sparse volumetric refinement. Given a
single RGB image, the model predicts a metric point map, depth map, normal
map, valid-pixel mask, and camera intrinsics in one forward pass.

Unlike standard VLMs, this model outputs dense geometry instead of text.

## Supported checkpoints

| HF repo (MLX) | Source checkpoint | Backbone |
| --- | --- | --- |
| `mlx-community/moge-3-vitl-mlx-fp32` | `Ruicheng/moge-3-vitl` | DINOv2 ViT-L/14 |
| `mlx-community/moge-3-vitg-mlx-fp32` | `Ruicheng/moge-3-vitg` | DINOv2 ViT-g/14 |

The MLX repos are produced from the official `model.pt` checkpoints with
`python -m mlx_vlm.models.moge3.convert`.

## Usage

```python
from mlx_vlm import load
from mlx_vlm.models.moge3.generate import MoGe3Predictor, read_image

model, processor = load("mlx-community/moge-3-vitl-mlx-fp32")
predictor = MoGe3Predictor(model, processor)

output = predictor.infer(read_image("image.jpg"), resolution_level=9)
points = output["points"]      # (H, W, 3) metric camera-space point map
depth = output["depth"]        # (H, W) metric depth map
mask = output["mask"]          # (H, W) valid-pixel mask
normal = output["normal"]      # (H, W, 3) normal map
intrinsics = output["intrinsics"]  # (3, 3) normalized camera intrinsics
```

Useful options: `num_tokens` / `resolution_level` (0-9) control the
inference resolution, `fov_x` pins the horizontal field of view in degrees,
`refine_steps` sets the number of sparse 3D refinement updates (default 3),
and `return_per_step=True` additionally returns the initial estimate and
every refinement step.

## Implementation notes

- The sparse 3D UNet refiner normally depends on FlexGEMM (Triton,
  CUDA-only). Here the submanifold convolution, mean pooling, and nearest
  upsample are re-implemented in pure MLX with sort + binary-search +
  gather, so the model runs on Apple silicon.
- Conv weights are stored channel-last. ``Model.sanitize`` relayouts
  ``ConvTranspose2d`` weights that are still in the torch ``(in, out, 2, 2)``
  layout (as in the first published MLX checkpoints) by comparing against
  the module's parameter shape, so both layouts load.
- The processor accepts ``(H, W, 3)`` or ``(B, H, W, 3)`` images. Integer
  images are scaled by their dtype range; float images must be in [0, 1].
- All image/feature resizes match `torch.nn.functional.interpolate`
  (`align_corners=False`), including the antialiased bilinear downscale in
  the encoder and the scale-factor bicubic position-embedding interpolation
  in DINOv2.
- On GPUs with matrix units (Apple M5 and later), MLX runs float32
  matmuls, convolutions, and attention at TF32 precision by default, which
  is roughly 3e-3 relative error. For full float32 geometry, set
  `MLX_ENABLE_TF32=0` in the environment before the first MLX matmul runs
  (the setting is latched on first use).
