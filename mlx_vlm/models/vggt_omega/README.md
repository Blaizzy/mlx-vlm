# VGGT-Omega (MLX)

Port of [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) (Meta AI and
Oxford VGG, CVPR 2026): feed-forward camera and depth reconstruction from a
sequence of images. One forward pass predicts, for every frame, the camera
(extrinsics and intrinsics), a depth map and a depth confidence map. The
text-aligned checkpoint also gives a language-aligned sequence embedding.

Unlike standard VLMs, this model outputs geometry instead of text.

## Supported checkpoints

| HF repo (MLX) | Source checkpoint | Resolution |
| --- | --- | --- |
| `mlx-community/VGGT-Omega-1B-512-bf16` | `facebook/VGGT-Omega` `vggt_omega_1b_512.pt` | 512 |

The other official checkpoints (`vggt_omega_1b_416_reproduce.pt`,
`vggt_omega_1b_256_text.pt`) use the same architecture and convert with the
same script. The official weights are gated; accept the license on
[facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) first.

```bash
python -m mlx_vlm.models.vggt_omega.convert \
    --hf-repo facebook/VGGT-Omega --checkpoint vggt_omega_1b_512.pt \
    --mlx-path VGGT-Omega-1B-512-bf16 --dtype bfloat16
```

The converter reads the `.pt` file without torch.

## Usage

```python
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.vggt_omega.generate import VGGTOmegaPredictor, read_video

model, processor = load("mlx-community/VGGT-Omega-1B-512-bf16")
predictor = VGGTOmegaPredictor(model, processor)

# Paths, URLs, PIL images or RGB arrays; the first image is the reference frame.
out = predictor.infer(["frame_000.png", "frame_001.png", "frame_002.png"])
# or: out = predictor.infer(read_video("clip.mp4", fps=1))
mx.eval(out)

out["extrinsics"]    # (S, 3, 4) camera-from-world, OpenCV convention
out["intrinsics"]    # (S, 3, 3) in pixels of the model input
out["depth"]         # (S, H, W)
out["depth_conf"]    # (S, H, W), >= 1
out["world_points"]  # (S, H, W, 3) unprojected depth
out["images"]        # (S, H, W, 3) model input in [0, 1]
```

`infer` builds one lazy graph, from image decoding to world points; the
work runs when an output is read. The processor mirrors the reference
`load_and_preprocess_images` (`mode="balanced"` or `"max_size"`,
`image_resolution`), as GPU ops.

From the command line, with an `.npz` of all outputs and a colored point
cloud filtered by confidence as in the reference demo:

```bash
python -m mlx_vlm.models.vggt_omega.generate \
    --model mlx-community/VGGT-Omega-1B-512-bf16 \
    --video clip.mp4 --fps 1 --output scene.npz --ply scene.ply
```

## Implementation notes

- Precision follows the reference: the aggregator runs in the weight dtype
  (bf16 matmuls and attention, as under CUDA autocast) with a float32
  residual stream and float32 norms, and the heads run in float32
  (`head_dtype` in `config.json`; the bf16 export upcasts them on load).
  GELU is computed in float32 and rounded once. In bf16 math, `1 + erf`
  cancels for negative inputs (45% off at x = -3, 0 at x = -4), and the
  error compounds over the 72 blocks.
- `kernels.py` has two fused Metal kernels: Q/K LayerNorm + 2D RoPE +
  layout for the attention inputs, and the float32 residual update + the
  next LayerNorm. Both run at memory bandwidth (~550 GB/s on M5 Max); as
  MLX ops, with or without `mx.compile`, the same work is 25x and 2-3x
  slower. At 16 frames they take a frame-attention block from 17.0 ms to
  9.8 ms, and the full forward pass is 1.26x (16 frames) to 1.35x (1 and
  8 frames) faster. Accuracy against the reference is unchanged.
- The released checkpoints store all-zero `qkv.bias_mask` buffers in the
  aggregator, so the reference runs those projections without bias. The
  converter folds each mask into its bias, which keeps this behavior.
- The DINOv3 encoder's RoPE `periods` are stored bf16-rounded in the
  checkpoint and are loaded, not recomputed. Both `periods` buffers stay
  float32 in the bf16 export.
- The dense head runs 2 frames per pass (`dense_frames_chunk_size`; the
  reference uses 8). The results do not change, and on M5 Max this was
  both faster and smaller in peak memory.
- Images get EXIF orientation applied (the reference ignores it). The
  bicubic resize matches PIL to within one uint8 level.
- On GPUs with matrix units (M5 and later), MLX runs float32 matmuls at
  TF32 precision by default. For full float32 heads, set
  `MLX_ENABLE_TF32=0` before the first MLX matmul runs.

## Accuracy and speed

Against the float32 PyTorch reference (8 frames, 688x384): the float32
port matches to ~1e-5 (TF32 off). The bf16 port is on par with the
reference's own bf16 autocast (run on CPU): depth median relative error
8.6e-4 (reference 1.2e-3), camera rotation within 0.04 degrees. Across
four scenes, bf16 rotation errors are 0.03 to 0.16 degrees.

bf16 on M5 Max, 688x384 frames, full forward pass:

| Frames | MLX | PyTorch MPS (bf16 autocast) | Peak memory (MLX) |
| --- | --- | --- | --- |
| 1 | 65 ms | 112 ms | 4.2 GB |
| 8 | 0.50 s | 1.22 s | 5.5 GB |
| 16 | 1.20 s | 3.02 s | 5.9 GB |
| 32 | 3.93 s | 8.72 s | 7.3 GB |

Global attention is quadratic in the number of frames and dominates long
sequences.

## License

The weights are under the FAIR Noncommercial Research License (see the
source repo); the MLX repos ship a copy.
