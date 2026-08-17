# YOLOv8 object detection (MLX)

Native MLX port of the Ultralytics YOLOv8n detection architecture, added to
`mlx-vlm` to serve as the `icon_detect` module of
[OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0). This is the
first YOLO port in `mlx-vlm`.

## Model Overview

- **Architecture**: YOLOv8n backbone (Darknet) + FPN/PANet neck + anchor-free
  Detect head with DFL (Distribution Focal Loss) box decoding.
- **Tensor layout**: handles the PyTorch `(B, C, H, W)` to MLX `(B, H, W, C)`
  conversion for all Conv2d weights.
- **Output**: `(B, 8400, 84)` — 8400 anchor points across 3 scales
  (P3/8, P4/16, P5/32) with 4 box coordinates + 80 class scores per point.

## Building blocks

| Block | PyTorch source | Purpose |
|---|---|---|
| `Conv` | `nn/modules/conv.py` | Conv2d + BatchNorm + SiLU (identity when `act=False`) |
| `Bottleneck` | `nn/modules/block.py` | Residual bottleneck (`x + cv2(cv1(x))`) |
| `C2f` | `nn/modules/block.py` | CSP bottleneck — 2 convs + `n` bottlenecks |
| `SPPF` | `nn/modules/block.py` | Spatial Pyramid Pooling — 3 sequential max-pools |
| `DFL` | `nn/modules/head.py` | 16-bin distribution → 4 box distances via fixed-weight matmul |
| `Detect` | `nn/modules/head.py` | Anchor-free detection head (box + class branches per scale) |

## Usage

### Load weights and run detection

```python
from mlx_vlm.models.yolov8 import YOLOv8, sanitize_yolo, load_weights

model = YOLOv8(nc=80)
mlx_weights = sanitize_yolo("yolov8n.safetensors")  # PyTorch -> MLX layout
load_weights(model, mlx_weights)
```

### Post-processing (decode + NMS)

```python
import mlx.core as mx
from mlx_vlm.models.yolov8 import make_anchors, dist2bbox, non_max_suppression

# model(x) returns (B, 4, 8400) distances and (B, 80, 8400) raw scores
dfl_out, scores = model(x)

# Generate per-anchor strides for the 3 detection scales (P3/8, P4/16, P5/32).
# For a 640x640 input the feature maps are 80x80, 40x40, 20x20 (8400 anchors).
anchor_feats = [
    mx.zeros((1, 80, 80, 64)),
    mx.zeros((1, 40, 40, 128)),
    mx.zeros((1, 20, 20, 256)),
]
anchors, strides = make_anchors(anchor_feats, [8, 16, 32])

# Decode to pixel-space xywh boxes using per-anchor strides
boxes = dist2bbox(dfl_out, anchors) * strides.reshape(1, 1, -1)

# Full prediction: [boxes, class scores] -> NMS
cls = mx.sigmoid(scores)
pred = mx.concatenate([boxes, cls], axis=1)  # (1, 84, 8400)
detections = non_max_suppression(pred)       # list of (N, 6) [x1, y1, x2, y2, score, class]
```

## Status

- **WIP**: foundational blocks, backbone, neck, and detection head are
  implemented and produce correct `(B, 8400, 84)` output shapes on random
  input.
- **Pending**: end-to-end OmniParser pipeline orchestrating YOLO detection
  (icon_detect) + Florence-2 captioning (icon_caption), model tests, and
  validation against the reference PyTorch model with real weights.

## Technical Details

- **Anchor-free**: the model predicts left/top/right/bottom distances from grid
  cell centers, not offsets from predefined anchor boxes.
- **fused weights**: `nn.Conv2d` layers use `bias=False`; BatchNorm absorbs the
  bias during inference, matching Ultralytics.
- **SPPF act=False**: `cv1` in `SPPF` disables the SiLU activation, matching
  `nn/modules/block.py`.