# SAM 3D Body for MLX

MLX port of [Meta's SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (arXiv:2603.15603) — single-image 3D human body mesh estimation using a DINOv3 vision backbone and MHR parametric body model.

> **Note:** Unlike SAM 3 / SAM 3.1 (segmentation), SAM 3D Body predicts a full 3D mesh (18,439 vertices, 36,874 faces) with 127 skeletal joints and 70 3D keypoints from a single RGB image. The entire pipeline — backbone, decoder, FK, skinning — runs in pure MLX with no PyTorch dependency at inference.

## What It Does

| Output | Shape | Description |
|--------|-------|-------------|
| Mesh vertices | (18439, 3) | Full body surface mesh |
| Skeleton | (127, 4, 4) | Joint world transforms (FK chain) |
| 3D keypoints | (70, 3) | Body + hand + face landmarks |
| Camera | (3,) | Weak-perspective (scale, tx, ty) |

The model takes a cropped person image (512×384), encodes it through a DINOv3-H+ ViT (32 layers, 1280d), decodes pose/shape tokens with a 6-layer transformer decoder, then runs the MHR body model (forward kinematics → blend shapes → pose correctives → linear blend skinning) to produce the final mesh.

## Quick Start

```python
from mlx_vlm.models.sam3d_body.generate import SAM3DPredictor

predictor = SAM3DPredictor.from_pretrained("/path/to/sam3d-mlx-weights")
```

## Single-Image Inference

```python
import numpy as np
from PIL import Image

image = np.array(Image.open("photo.jpg").convert("RGB"))
result = predictor.predict(image, bbox=[100, 50, 400, 500])

# result["pred_vertices"]      -> (18439, 3) mesh vertices
# result["pred_keypoints_3d"]  -> (70, 3) 3D keypoints
# result["pred_camera"]        -> (3,) weak-perspective camera
```

## Mesh Export

```python
from mlx_vlm.models.sam3d_body.estimator import write_obj

write_obj(result["pred_vertices"], faces, "output.obj")
```

## Single-Image Overlay

Two flavors of overlay on the original frame — skeleton-only (pure OpenCV,
no extra deps) and full photorealistic mesh (requires `pyrender` + `trimesh`).

```python
import cv2
from mlx_vlm.models.sam3d_body.overlay import (
    draw_skeleton_overlay, render_mesh_overlay, load_faces,
)

frame_bgr = cv2.imread("photo.jpg")

# Skeleton only (fast, no extra deps)
skel_bgr = draw_skeleton_overlay(result, frame_bgr)
cv2.imwrite("photo_skeleton.jpg", skel_bgr)

# Full mesh overlay (requires: pip install pyrender trimesh)
faces = load_faces("/path/to/sam3d-mlx-weights")
mesh_bgr = render_mesh_overlay(result, frame_bgr, faces)
cv2.imwrite("photo_mesh.jpg", mesh_bgr)
```

## Video Pipeline

Frame-by-frame body estimation with skeleton overlay rendering:

```bash
python -m mlx_vlm.models.sam3d_body.video --input pitch.mp4 --output pitch_overlay.mp4
```

## CLI

```bash
# Single image → OBJ mesh
python -m mlx_vlm.models.sam3d_body.generate --image photo.jpg --output mesh.obj

# With bounding box
python -m mlx_vlm.models.sam3d_body.generate --image photo.jpg --bbox 100,50,400,500 --output mesh.obj

# Custom weights directory
python -m mlx_vlm.models.sam3d_body.generate --image photo.jpg --weights /path/to/weights/ --output mesh.obj

# Save 3D keypoints alongside mesh
python -m mlx_vlm.models.sam3d_body.generate --image photo.jpg --output mesh.obj --save-keypoints
```

## Architecture

```
Image (512×384 RGB)
  │
  ▼
DINOv3-H+ Backbone (32 layers, 1280d, 20 heads, 2D axial RoPE)
  │
  ├── storage tokens (4)
  │
  ▼
Prompt Encoder (70 keypoint embeddings + hand box embedding)
  │
  ▼
Transformer Decoder (6 layers, 1024d, 8 heads)
  │    cross-attention to backbone features
  │    + ray conditioning (camera intrinsics → 1379-ch ray map → 1×1 conv)
  │
  ├── pose token ──► MHR Head (FFN → 519 params)
  │                     │
  │                     ├── body rotation: 33 joints × 6D cont.
  │                     ├── shape: 45 betas
  │                     ├── face: 72 expression coeffs
  │                     └── hand: 54 PCA coeffs (27 per hand)
  │                     │
  │                     ▼
  │                  MHR Body Model (pure MLX)
  │                     ├── Parameter transform (889×249 matrix)
  │                     ├── Forward kinematics (127 joints, quaternion prerotations)
  │                     ├── Blend shapes (45 shape + 72 face vectors)
  │                     ├── Pose correctives (sparse predictor → dense)
  │                     └── Linear blend skinning (51,337 sparse weights)
  │                     │
  │                     ▼
  │                  Skinned vertices (18439, 3) + skeleton (127, 4×4)
  │
  ├── camera token ──► Camera Head (FFN → 3: scale, tx, ty)
  │
  └── keypoint tokens ──► 3D Keypoint regression (70 joints)
```

### Components

| Component | Description | Weight Prefix |
|-----------|-------------|---------------|
| Vision Encoder | DINOv3-H+ ViT, 32 blocks, 2D axial RoPE | `backbone.*` |
| Prompt Encoder | Keypoint/box prompt embeddings | `prompt_encoder.*` |
| Decoder | 6-layer transformer with cross-attention | `decoder.*` |
| Ray Conditioning | 1×1 conv from camera rays to embed_dim | `ray_condition_embed.*` |
| MHR Pose Head | FFN projecting decoder token → 519 params | `head_pose.proj.*` |
| MHR Body Model | FK + blend shapes + skinning (pure MLX) | `head_pose.body_model.*` |
| Camera Head | FFN predicting weak-perspective camera | `head_camera.*` |
| Keypoint MLPs | 2D/3D keypoint position embeddings | `keypoint_posemb_linear.*`, `keypoint3d_posemb_linear.*` |

### MHR Body Model Detail

The body model is a pure MLX reimplementation of the original TorchScript JIT model. No PyTorch at inference.

| Stage | What it does | Key buffers |
|-------|-------------|-------------|
| Parameter transform | Maps 519 network outputs → 889 internal params via a learned linear transform | `parameter_transform` (889×249) |
| Forward kinematics | Builds 127-joint skeleton from local rotations + quaternion prerotations | `joint_translation_offsets`, `joint_prerotations`, `joint_parents` |
| Blend shapes | Adds shape (45 PCA) + face (72 expression) deformations to rest mesh | `shape_vectors` (45×18439×3), `face_expressions.shape_vectors` (72×18439×3) |
| Pose correctives | Sparse predictor → dense vertex corrections from joint rotations | Sparse indices (53,136 entries) + dense layer (55317×3000) |
| LBS | Weighted sum of joint transforms per vertex | `skin_weights_flattened` (51,337 entries) |

## Benchmarks (Apple Silicon)

Measured on M3 Max (36 GB), float16 precision.

| Metric | Value |
|--------|-------|
| Single image | ~490ms |
| Backbone (DINOv3 32L) | ~280ms |
| Decoder + heads | ~60ms |
| MHR body model (FK + skinning) | ~150ms |
| Model parameters | ~720M |
| Vertex accuracy vs PyTorch | < 0.001mm |

Video processing at ~2 FPS sustained on M3 Max.

## Weight Conversion

SAM 3D Body weights come as a PyTorch `.ckpt` plus a TorchScript JIT `.pt` file for the MHR body model. The converter handles both:

```bash
python -m mlx_vlm.models.sam3d_body.convert_weights \
    --checkpoint /path/to/model.ckpt \
    --mhr-model /path/to/assets/mhr_model.pt \
    --output /path/to/sam3d-mlx-weights/
```

Key conversions:
- **QKV splitting**: Fused `qkv.weight`/`qkv.bias` → separate `q_proj`, `k_proj`, `v_proj`
- **Conv2d transposition**: PyTorch `(O,I,H,W)` → MLX `(O,H,W,I)`
- **JIT model extraction**: `torch.jit.load` → iterate `named_buffers()` + `named_parameters()`
- **Dtype**: bfloat16 → float16, int64 → int32
- **Output**: `model.safetensors` (or sharded) + `config.json`

## File Structure

```
mlx_vlm/models/sam3d_body/
├── __init__.py            # Module exports (Model, ModelConfig, VisionModel, LanguageModel)
├── config.py              # SAM3DConfig, VisionConfig, TextConfig dataclasses
├── model.py               # SAM3DBody — top-level forward, weight loading, sanitize()
├── backbone.py            # DINOv3Backbone — ViT-H+ (32 blocks, 1280d, 2D RoPE)
├── rope.py                # DINOv3RoPE — 2D axial RoPE with learned periods
├── layers.py              # LayerNorm32, SwiGLU, LayerScale
├── prompt_encoder.py      # Keypoint/box prompt → embeddings
├── decoder.py             # PromptableDecoder wrapper
├── transformer.py         # DecoderFFN, transformer decoder layers
├── mhr_head.py            # MHRHead — pose FFN → parameter extraction → body model
├── mhr_body.py            # MHRBodyModel — FK, blend shapes, skinning (pure MLX)
├── mhr_utils.py           # Rotation math (rot6d, euler, quaternion), parameter mapping
├── camera.py              # Perspective projection
├── batch_prep.py          # Crop, resize, ImageNet normalize, CLIFF conditioning
├── estimator.py           # SAM3DBodyEstimator — preprocessing + inference + OBJ export
├── generate.py            # SAM3DPredictor — mlx-vlm from_pretrained/predict API
├── video.py               # Video pipeline with skeleton overlay rendering
├── overlay.py             # Single-image skeleton + pyrender mesh overlay helpers
├── convert_weights.py     # PyTorch .ckpt + JIT .pt → safetensors converter
├── vision.py              # VisionModel stub (wraps backbone for mlx-vlm compat)
└── language.py            # LanguageModel stub (raises NotImplementedError)
```

### mlx-vlm Integration

This package follows mlx-vlm model conventions:
- `Model = SAM3DBody`, `ModelConfig = SAM3DConfig` aliases for the framework loader
- `VisionModel` wraps the DINOv3 backbone; `LanguageModel` is a stub (SAM 3D Body is vision-only)
- `sanitize()` static method for mlx-vlm's weight loading path
- `model_type: "sam3d_body"` in config, intended for mlx-vlm's `MODEL_REMAPPING`

## Future Work

- **Hand pose refinement pass.** The upstream model ships a second decoder stage
  that refines each hand independently using detected hand-box crops
  (`decoder_hand.*`, `head_pose_hand.*`, etc. in the checkpoint). This port
  filters those weights out and runs the body-only pass, which is accurate for
  full-body poses but leaves the fingers at the body head's coarser prediction.
  A follow-up PR will wire the hand refinement decoder.
- **Hand box detection head.** The body decoder already emits predicted hand
  bounding boxes (`hand_cls_embed`, `bbox_embed`) but they are unused until
  the hand pass lands.

## License

The original SAM 3D Body model weights are released by Meta. The MLX port code follows the same license as [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). See the upstream repository for model weight license details.
