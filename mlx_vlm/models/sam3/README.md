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

## Image: Object Detection

Detect objects by text prompt — returns bounding boxes and confidence scores:

```python
result = predictor.predict(image, text_prompt="a dog")

for i in range(len(result.scores)):
    x1, y1, x2, y2 = result.boxes[i]
    print(f"[{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

## Image: Instance Segmentation

Same call — per-instance masks are returned alongside boxes:

```python
result = predictor.predict(image, text_prompt="a person")

# result.boxes   -> (N, 4) xyxy bounding boxes, scaled to image size
# result.masks   -> (N, H, W) binary segmentation masks
# result.scores  -> (N,) confidence scores
```

### Overlay Masks on Image

```python
import numpy as np

result = predictor.predict(image, text_prompt="a cat")

overlay = np.array(image).copy()
colors = [(30, 120, 255), (255, 80, 30), (30, 200, 30)]

W, H = image.size
for i in range(len(result.scores)):
    mask = result.masks[i]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)))
    binary = mask > 0
    color = np.array(colors[i % len(colors)])
    overlay[binary] = (overlay[binary] * 0.5 + color * 0.5).astype(np.uint8)

Image.fromarray(overlay).save("segmentation_output.png")
```

## Image: Box-Guided Detection

Pass bounding box prompts to guide detection to specific regions:

```python
import numpy as np

# Box covering a region of interest in xyxy pixel coordinates
boxes = np.array([[100, 50, 400, 350]])
result = predictor.predict(image, text_prompt="a cat", boxes=boxes)
```

## Image: Semantic Segmentation

Access the dense semantic segmentation output alongside instance predictions:

```python
import mlx.core as mx

inputs = processor.preprocess_image(image)
text_inputs = processor.preprocess_text("a cat")

outputs = model.detect(
    mx.array(inputs["pixel_values"]),
    mx.array(text_inputs["input_ids"]),
    mx.array(text_inputs["attention_mask"]),
)
mx.eval(outputs)

# Instance masks: (B, 200, 288, 288)
pred_masks = outputs["pred_masks"]

# Semantic segmentation: (B, 1, 288, 288)
semantic_seg = outputs["semantic_seg"]
```

## Image: Batched Inference

Process multiple images in a single forward pass:

```python
import mlx.core as mx
import numpy as np

images = [Image.open("photo1.jpg"), Image.open("photo2.jpg")]

# Stack pixel values into a batch
pixel_values = mx.array(np.stack([
    processor.preprocess_image(img)["pixel_values"][0] for img in images
]))  # (B, 1008, 1008, 3)

# Encode text once (shared across the batch)
text_inputs = processor.preprocess_text("a cat")
input_ids = mx.array(np.tile(text_inputs["input_ids"], (len(images), 1)))
attention_mask = mx.array(np.tile(text_inputs["attention_mask"], (len(images), 1)))

outputs = model.detect(pixel_values, input_ids, attention_mask)
mx.eval(outputs)

# outputs["pred_logits"]: (B, 200)
# outputs["pred_boxes"]:  (B, 200, 4)
# outputs["pred_masks"]:  (B, 200, 288, 288)
for i in range(len(images)):
    scores = 1 / (1 + np.exp(-np.array(outputs["pred_logits"][i])))
    print(f"Image {i}: top score = {scores.max():.2f}")
```


## Video: Tracking (CLI)

Track objects in a video from the command line:

```bash
python -m mlx_vlm.models.sam3.generate --video input.mp4 --prompt "a car"
```

Output saves to `input_tracked.mp4` by default. Full options:

```bash
python -m mlx_vlm.models.sam3.generate \
    --video input.mp4 \
    --prompt "a person" \
    --output output.mp4 \
    --threshold 0.2 \
    --every 4 \
    --nms-thresh 0.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Input video path |
| `--prompt` | *(required)* | Text prompt for detection |
| `--output` | `<input>_tracked.mp4` | Output video path |
| `--model` | `facebook/sam3` | Model path or HF repo |
| `--threshold` | `0.15` | Detection score threshold |
| `--every` | `2` | Run detection every N frames |
| `--nms-thresh` | `0.5` | NMS IoU threshold |

## Video: Tracking (Python)

```python
import cv2
from mlx_vlm.models.sam3.generate import Sam3VideoPredictor

cap = cv2.VideoCapture("video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()

video_predictor = Sam3VideoPredictor(model, processor, score_threshold=0.15)
video_predictor.set_video(frames)
video_predictor.add_text_prompt("a car", frame_idx=0)

results = video_predictor.propagate()
for r in results:
    print(f"Frame {r.frame_idx}: {len(r.object_ids)} objects tracked")
```

### Video: Batched Frame Processing

For maximum throughput, process multiple video frames at once using the low-level API:

```python
import cv2
import mlx.core as mx
import numpy as np

cap = cv2.VideoCapture("video.mp4")
batch_size = 4

# Encode text once, reuse for all frames
text_inputs = processor.preprocess_text("a car")
input_ids = mx.array(text_inputs["input_ids"])
attention_mask = mx.array(text_inputs["attention_mask"])
inputs_embeds, attention_mask = model.get_text_features(input_ids, attention_mask)

while cap.isOpened():
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(processor.preprocess_image(pil)["pixel_values"][0])

    if not frames:
        break

    pixel_values = mx.array(np.stack(frames))  # (B, 1008, 1008, 3)
    B = len(frames)

    outputs = model.detect(
        pixel_values,
        attention_mask=mx.tile(attention_mask, (B, 1)),
        inputs_embeds=mx.tile(inputs_embeds, (B, 1, 1)),
    )
    mx.eval(outputs)

    # Process each image in the batch
    for i in range(B):
        scores = 1 / (1 + np.exp(-np.array(outputs["pred_logits"][i])))
        print(f"  {scores[scores > 0.3].shape[0]} detections")

cap.release()
```

> **Tip:** Use `--every N` in the CLI to skip frames — this gives a real Nx speedup since it avoids the expensive ViT pass entirely. Batching processes more frames but doesn't improve per-frame speed.

See [`examples/sam3_demo.ipynb`](../../../examples/sam3_demo.ipynb) for a full interactive notebook demo.

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
