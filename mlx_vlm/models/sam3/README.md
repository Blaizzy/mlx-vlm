# SAM3 (Segment Anything Model 3) for MLX

MLX port of [Meta's SAM3](https://github.com/facebookresearch/sam3) — an open-vocabulary detection, segmentation, and video tracking model (~860M parameters).

> **Note:** SAM3 is not a generative VLM. It outputs bounding boxes and segmentation masks, not text. It uses a custom inference pipeline (`generate.py`) instead of `mlx_vlm.generate()`.

## Quick Start

```python
from PIL import Image
from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

# Load model (downloads ~3.4 GB on first run)
model_path = get_model_path("facebook/sam3")
model = load_model(model_path)
processor = Sam3Processor.from_pretrained(str(model_path))

image = Image.open("photo.jpg")
predictor = Sam3Predictor(model, processor, score_threshold=0.3)
```

### Object Detection

Detect objects by text prompt — returns bounding boxes and confidence scores:

```python
result = predictor.predict(image, text_prompt="a dog")

for i in range(len(result.scores)):
    x1, y1, x2, y2 = result.boxes[i]
    print(f"[{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

### Instance Segmentation

Same call — masks are returned alongside boxes:

```python
result = predictor.predict(image, text_prompt="a person")

# result.boxes   -> (N, 4) xyxy bounding boxes, scaled to image size
# result.masks   -> (N, H, W) binary segmentation masks
# result.scores  -> (N,) confidence scores

# Overlay mask on image
import numpy as np
mask = result.masks[0]  # first detection
overlay = np.array(image).copy()
overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
```

### Video Tracking

Track objects across video frames using text, point, or box prompts:

```python
import cv2
from mlx_vlm.models.sam3.generate import Sam3VideoPredictor

# Load video frames
cap = cv2.VideoCapture("video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()

# Set up tracker
video_predictor = Sam3VideoPredictor(model, processor, score_threshold=0.15)
video_predictor.set_video(frames)

# Initialize with any prompt type on a keyframe
video_predictor.add_text_prompt("a car", frame_idx=0)

# Propagate through all frames
results = video_predictor.propagate()
for r in results:
    # r.frame_idx, r.masks (N_obj, H, W), r.scores (N_obj,), r.object_ids
    print(f"Frame {r.frame_idx}: {len(r.object_ids)} objects tracked")
```

## Architecture

```
Image (1008x1008)
  |
  v
ViT Backbone (32 layers, 1024d, windowed + global attention, 2D RoPE)
  |
  v
FPN Neck (4 scales: 288x288, 144x144, 72x72, 36x36)
  |                                |
  v                                v
DETR Encoder (6 layers)     Tracker Neck (separate FPN)
  + text cross-attention         |
  |                              v
  v                        SAM2 Tracker
DETR Decoder (6 layers)      Memory Attention
  200 queries                 Memory Encoder
  Box refinement              Mask Decoder
  Presence token
  BoxRPB
  |
  +---> DotProductScoring --> pred_logits
  +---> BoxHead           --> pred_boxes (xyxy)
  +---> MaskDecoder/FPN   --> pred_masks (288x288)
```

### Components

| Component | Description | Weight Prefix |
|-----------|-------------|---------------|
| Vision Encoder | ViT-L + FPN | `detector_model.vision_encoder.*` |
| Text Encoder | CLIP (24L, 1024d) | `detector_model.text_encoder.*` |
| DETR Encoder | 6-layer pre-norm transformer with text cross-attn | `detector_model.detr_encoder.*` |
| DETR Decoder | 6-layer post-norm, 200 queries, box refinement, BoxRPB | `detector_model.detr_decoder.*` |
| Geometry Encoder | Encodes box/point prompts | `detector_model.geometry_encoder.*` |
| Mask Decoder | Pixel decoder + instance projection | `detector_model.mask_decoder.*` |
| Dot Product Scoring | Text-query dot product classifier | `detector_model.dot_product_scoring.*` |
| Tracker | SAM2-style memory-based tracker | `tracker_model.*` |
| Tracker Neck | Separate FPN for tracker | `tracker_neck.*` |

## File Structure

```
mlx_vlm/models/sam3/
├── __init__.py           # Module exports
├── config.py             # All config dataclasses
├── sam3.py               # Main Model class + sanitize
├── vision.py             # ViT backbone + FPN neck
├── text_encoder.py       # CLIP text encoder
├── encoder.py            # DETR transformer encoder (pre-norm)
├── decoder.py            # DETR transformer decoder (post-norm, BoxRPB)
├── geometry.py           # Geometry encoder (box/point prompts)
├── segmentation.py       # Mask decoder + dot product scoring
├── position.py           # Sinusoidal + 2D RoPE position encodings
├── sam_components.py     # SAM prompt encoder, mask decoder, TwoWayTransformer
├── tracker.py            # Memory encoder, memory attention, tracker model
├── generate.py           # Inference pipeline (Sam3Predictor, Sam3VideoPredictor)
└── processing_sam3.py    # Image/text preprocessing
```
