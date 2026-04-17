# RF-DETR for MLX

Real-time detection transformer ([RF-DETR](https://github.com/roboflow/rf-detr), ICLR 2026) ported to Apple Silicon via MLX. Supports object detection and instance segmentation on COCO 80 classes.

## Quick Start

```python
from pathlib import Path
from PIL import Image
from mlx_vlm.utils import load_model
from mlx_vlm.models.rfdetr.processing_rfdetr import RFDETRProcessor
from mlx_vlm.models.rfdetr.generate import RFDETRPredictor

# Load model (uses the standard mlx-vlm loader)
model = load_model(Path("rfdetr-base-mlx"))
processor = RFDETRProcessor.from_pretrained("rfdetr-base-mlx")
predictor = RFDETRPredictor(model, processor, score_threshold=0.3, nms_threshold=0.5)

# Run detection
result = predictor.predict(Image.open("image.jpg"))

for name, score, box in zip(result.class_names, result.scores, result.boxes):
    print(f"{name}: {score:.2f} [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

## Convert Weights

One-time conversion from the official Roboflow checkpoints. Requires `torch` and `safetensors` (not needed at inference).

```bash
# Detection
python -m mlx_vlm.models.rfdetr.convert --variant base --output ./rfdetr-base-mlx
python -m mlx_vlm.models.rfdetr.convert --variant small --output ./rfdetr-small-mlx
python -m mlx_vlm.models.rfdetr.convert --variant large --output ./rfdetr-large-mlx

# Segmentation (detection + instance masks)
python -m mlx_vlm.models.rfdetr.convert --variant seg-small --output ./rfdetr-seg-small-mlx
python -m mlx_vlm.models.rfdetr.convert --variant seg-large --output ./rfdetr-seg-large-mlx
```

Each output directory contains `config.json`, `preprocessor_config.json`, and `model.safetensors`. No PyTorch or rfdetr dependency at runtime.

## Available Variants

| Variant | Task | Resolution | Params | Latency (M4 Max) |
|---------|------|-----------|--------|----------|
| `base` | Detection | 560 | ~32M | ~33ms |
| `small` | Detection | 512 | ~32M | - |
| `large` | Detection | 704 | ~128M | - |
| `seg-small` | Detection + Segmentation | 384 | ~34M | ~88ms |
| `seg-large` | Detection + Segmentation | 480 | ~130M | - |

## Segmentation

Segmentation models output per-instance binary masks alongside boxes:

```python
result = predictor.predict(image)

# result.boxes   - (N, 4) xyxy pixel coordinates
# result.scores  - (N,) confidence scores
# result.labels  - (N,) COCO class indices
# result.masks   - (N, H, W) binary uint8 masks (or None for detection-only)
```

## Filtering

Exclude unwanted classes or adjust thresholds:

```python
predictor = RFDETRPredictor(
    model, processor,
    score_threshold=0.3,     # minimum confidence
    nms_threshold=0.5,       # IoU threshold for NMS
    exclude_classes=["couch", "potted plant"],  # filter by name
)
```

## Architecture

```
Image (HxW) --> DINOv2-small (windowed attention, 12 layers)
            --> MultiScaleProjector (C2f block, P4)
            --> Two-Stage Encoder (top-K query selection)
            --> Decoder (3-4 layers, deformable cross-attention)
            --> Detection Head (class + bbox)
            --> [Segmentation Head] (optional, depthwise conv + einsum masks)
```

## File Structure

```
mlx_vlm/models/rfdetr/
  config.py               # Dataclass configs
  vision.py               # DINOv2 backbone + C2f projector
  transformer.py           # Encoder (two-stage) + Decoder (deformable attention)
  segmentation.py          # SegmentationHead (mask prediction)
  rfdetr.py               # Main Model + weight sanitization
  generate.py             # RFDETRPredictor + postprocessing + NMS
  processing_rfdetr.py    # Image preprocessing + COCO class names
  convert.py              # PyTorch checkpoint converter
  language.py             # Stub (framework compatibility)
```

## Reference

- [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2511.09554) (ICLR 2026)
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
