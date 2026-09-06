# DINOv2 (MLX)

Channel-last MLX port of the [DINOv2](https://github.com/facebookresearch/dinov2)
vision transformer. This package currently holds only the shared backbone and
its architecture presets. The standalone model is not ported yet: there is no
`Model` class, so `facebook/dinov2-*` checkpoints cannot be loaded through
`mlx_vlm.load`, and register tokens, masking and the DINO heads are not
implemented.

## Used by

| Model | Backbones |
| --- | --- |
| `video_depth_anything` | ViT-S/14, ViT-B/14, ViT-L/14 |
| `moge3` | ViT-L/14, ViT-g/14 |
