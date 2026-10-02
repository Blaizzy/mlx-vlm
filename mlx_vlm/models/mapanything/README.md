# MapAnything (MLX)

Port of [MapAnything](https://github.com/facebookresearch/map-anything)
(Meta and CMU, 3DV 2026): universal feed-forward metric 3D reconstruction.
From one or more images, plus any optional calibration, depth and camera
poses, one forward pass predicts for every view the metric point map, ray
directions, depth, camera pose and intrinsics, a confidence map and a
non-ambiguity mask.

The model is a DINOv2 encoder (the shared `mlx_vlm.models.dinov2` backbone),
encoders for the optional geometric inputs, a multi-view transformer that
alternates attention across all views and within each view, a DPT dense
head (the shared `mlx_vlm.models.dpt` fusion blocks) and pose and
metric-scale heads.

## Checkpoints

| Source repo | License | Encoder / transformer |
| --- | --- | --- |
| `facebook/map-anything` | CC-BY-NC 4.0 | DINOv2 ViT-g/14 (24 blocks) / 16 layers, dim 1536 |
| `facebook/map-anything-apache` | Apache 2.0 | same as above |
| `facebook/map-anything-v1` | CC-BY-NC 4.0 | DINOv2 ViT-L/14 / 24 layers, dim 768 |
| `facebook/map-anything-apache-v1` | Apache 2.0 | same as above |

The official repos ship PyTorch-layout float32 weights without a
`model_type`. Convert one to an MLX directory (float16 trunk, float32
heads; 2.6 GB for the current release):

```bash
python -m mlx_vlm.models.mapanything.convert \
    --hf-path facebook/map-anything --mlx-path map-anything-fp16
```

`generate.load` (and the command line below) also accept an official repo
directly and convert it in memory.

## Usage

```python
import mlx.core as mx
from mlx_vlm import load

model, processor = load("map-anything-fp16")

views = processor.load_images("path/to/images")   # folder or list of images
predictions = model.infer(views)                   # lazy MLX arrays
mx.eval(predictions)

for pred in predictions:
    pred["pts3d"]            # (1, H, W, 3) world points (first view's frame)
    pred["depth_z"]          # (1, H, W, 1) metric z-depth
    pred["camera_poses"]     # (1, 4, 4) OpenCV cam2world
    pred["intrinsics"]       # (1, 3, 3)
    pred["mask"]             # (1, H, W, 1) valid pixels
```

### Geometric inputs

Any view can add calibration, depth and pose, in any combination:

```python
views = processor.preprocess([
    {"img": image0, "intrinsics": K0, "depth_z": depth0,
     "camera_poses": pose0, "is_metric_scale": True},
    {"img": image1, "intrinsics": K1},                     # calibration only
    {"img": image2, "ray_directions": rays2, "depth_z": depth2},
    {"img": image3, "camera_poses": (quats3, trans3)},    # needs a posed view 0
])
predictions = model.infer(views)
```

- `img`: (H, W, 3) uint8, float in [0, 1], a PIL image or a path.
- `intrinsics` (3, 3) *or* `ray_directions` (H, W, 3), not both.
- `depth_z` (H, W) needs `intrinsics` or `ray_directions`.
- `camera_poses`: (4, 4) OpenCV cam2world or `(quats_xyzw, trans)`. When any
  view is posed, view 0 must be posed too; it defines the world frame.
- `is_metric_scale` (default True) says whether depth and pose translations
  are metric.

`preprocess` resizes every input to the shared model resolution (images with
Pillow's filters, depth with nearest sampling, intrinsics rescaled and
shifted). `model.infer` also takes views that are already at model
resolution, with `img` as (B, H, W, 3) normalized arrays.

### Outputs

Per view: `pts3d`, `pts3d_cam`, `ray_directions`, `depth_along_ray`,
`depth_z`, `cam_trans`, `cam_quats`, `camera_poses`, `intrinsics`,
`metric_scaling_factor`, `conf`, `non_ambiguous_mask`,
`non_ambiguous_mask_logits`, `mask` and `img_no_norm` (the input image in
[0, 1]), with the shapes of the reference `infer`.

`infer` options match the reference: `apply_mask`, `mask_edges`
(`edge_normal_threshold`, `edge_depth_threshold`), `apply_confidence_mask`
(`confidence_percentile`), `ignore_calibration_inputs`,
`ignore_depth_inputs`, `ignore_pose_inputs`, `ignore_depth_scale_inputs`,
`ignore_pose_scale_inputs` and `use_multiview_confidence`
(`multiview_conf_depth_abs_thresh`, `multiview_conf_depth_rel_thresh`).

### Command line

```bash
python -m mlx_vlm.models.mapanything.generate \
    --model facebook/map-anything --images path/to/images --output out
```

Writes `out/predictions.npz` (every output of every view) and
`out/points.ply` (masked points, colored by the images).

## Implementation notes

- Lazy end to end: after the images are decoded, preprocessing, the network
  and all post-processing (intrinsics fit, edge masks, confidence
  percentiles, multi-view depth consistency with the frustum-overlap test)
  are MLX ops on the GPU. `infer` never synchronizes; evaluate the returned
  arrays when you need them.
- Precision: the encoder and multi-view transformer run their matmuls and
  attention in the weight dtype (float16 by default, which is what the
  reference's mixed precision uses on Apple GPUs) with a float32 residual
  stream, norms and activations. The patch embedding, geometric input
  encoders and heads run in float32, as in the reference.
- On Apple M5 and later, MLX runs float32 matmuls and convolutions at TF32
  precision by default, and that rounding in the heads becomes the largest
  error. Set `MLX_ENABLE_TF32=0` before the first MLX matmul for full float32
  heads; it costs a few milliseconds. The command line does this unless
  `--tf32` is passed.
- `memory_efficient_inference` / `minibatch_size` run the full-resolution
  DPT head a few images at a time (default `dense_head_chunk_size: 4` in the
  config), which bounds its memory. The trunk precision comes from the
  weights (`--dtype` at conversion) instead of `use_amp` / `amp_dtype`.
- The edge mask takes `arccos` of clipped cosines. The reference feeds
  cosines that round above 1 to `arccos` and gets NaN; this changes about
  0.02% of its normal-edge pixels.
- Image resizing reproduces Pillow's LANCZOS / BICUBIC filters and two-pass
  uint8 rounding in float32; about 0.03% of pixel values differ from Pillow
  by one level.

## Validation

Against the PyTorch reference (float32 on CPU), with float32 weights and
`MLX_ENABLE_TF32=0`, 4 to 5 views of 518x294:

| Output | Median relative error |
| --- | --- |
| points, depth, confidence, mask logits | ~1e-6 |
| camera poses, intrinsics, metric scale | ~1e-6 |
| final mask (pixels that differ) | < 0.02% |

This holds for image-only input, for mixed calibration / depth / pose
inputs, for multi-view confidence, and for the v1 checkpoints. With the
float16 trunk, depth after scale normalization is within 5e-5 to 2e-4
(median, relative) of float32, like the reference's mixed precision on MPS
(6e-5 to 3e-4).

Timings on an M5 Max (float16 trunk, 518x294 views; another GPU app was
running, so treat these as relative):

| Views | MLX | PyTorch MPS (mixed precision) |
| --- | --- | --- |
| 1 | 0.06 s | 0.13 s |
| 4 | 0.23 s | 0.47 s |
| 8 | 0.48 s | 1.73 s |
| 16 | 1.10 s | 5.27 s |
| 32 | 2.89 s | 11.3 s |
