# SAM 3.1 (Segment Anything Model 3.1) for MLX

MLX port of [Meta's SAM 3.1](https://github.com/facebookresearch/sam3) — extends SAM 3 with **Object Multiplex** for faster multi-object tracking (~7x at 128 objects).

> **Note:** SAM 3.1 shares the same detection pipeline as SAM 3 but adds a triple-head FPN, multiplex mask decoder (16 objects simultaneously), and decoupled memory attention for tracking.

## What's New in SAM 3.1

| Component | SAM 3 | SAM 3.1 |
|-----------|-------|---------|
| FPN Neck | 2 heads, 4 scales | **3 heads** (det, interactive, propagation), **3 scales** |
| Tracker Mask Decoder | 1 object at a time | **16 objects simultaneously** (MultiplexMaskDecoder) |
| Memory Attention | Standard transformer | **Decoupled** with image cross-attention + RoPE |
| Tracker | Single decoder | **Dual decoder** (interactive + propagation) |
| Detection scores | 0.87, 0.82 (cats) | **0.90, 0.86** (cats) — retrained weights |

## Quick Start

```python
from PIL import Image
from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor

model_path = get_model_path("mlx-community/sam3.1-bf16")
model = load_model(model_path)
processor = Sam31Processor.from_pretrained(str(model_path))
predictor = Sam3Predictor(model, processor, score_threshold=0.3)
```

## Object Detection

```python
image = Image.open("photo.jpg")
result = predictor.predict(image, text_prompt="a dog")

for i in range(len(result.scores)):
    x1, y1, x2, y2 = result.boxes[i]
    print(f"[{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

## Instance Segmentation

```python
result = predictor.predict(image, text_prompt="a person")

# result.boxes   -> (N, 4) xyxy bounding boxes
# result.masks   -> (N, H, W) binary segmentation masks
# result.scores  -> (N,) confidence scores

import numpy as np
overlay = np.array(image).copy()
W, H = image.size
for i in range(len(result.scores)):
    mask = result.masks[i]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)))
    binary = mask > 0
    overlay[binary] = (overlay[binary] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
```

## Box-Guided Detection

```python
import numpy as np
boxes = np.array([[100, 50, 400, 350]])
result = predictor.predict(image, text_prompt="a cat", boxes=boxes)
```

## CLI

SAM 3.1 uses the same CLI as SAM 3:

```bash
# Object detection
python -m mlx_vlm.models.sam3.generate --task detect --image photo.jpg --prompt "a cat" --model mlx-community/sam3.1-bf16

# Instance segmentation
python -m mlx_vlm.models.sam3.generate --image photo.jpg --prompt "a cat" --model mlx-community/sam3.1-bf16

# Video tracking
python -m mlx_vlm.models.sam3.generate --task track --video input.mp4 --prompt "a car" --model mlx-community/sam3.1-bf16

# Real-time webcam
python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person" --model mlx-community/sam3.1-bf16 --resolution 224
```

## Architecture

```
Image (1008x1008)
  |
  v
ViT Backbone (32 layers, 1024d, windowed + global attention, 2D RoPE)
  |
  v
TriViTDetNeck (3 parallel FPN heads, 3 scales: 288x288, 144x144, 72x72)
  |                    |                      |
  v                    v                      v
Detection FPN    Interactive FPN        Propagation FPN
  |                    |                      |
  v                    v                      v
DETR Encoder     Interactive SAM        Multiplex Tracker
  + Decoder        Mask Decoder           16 objects
  200 queries       (clicks/boxes)         simultaneously
  |
  +---> DotProductScoring --> pred_logits
  +---> BoxHead           --> pred_boxes (xyxy)
  +---> MaskDecoder/FPN   --> pred_masks (288x288)
```

### Components

| Component | Description | Weight Prefix |
|-----------|-------------|---------------|
| Vision Encoder | ViT-L + **TriViTDetNeck** (3 heads) | `detector_model.vision_encoder.*` |
| Text Encoder | CLIP (24L, 1024d) | `detector_model.text_encoder.*` |
| DETR Encoder | 6-layer pre-norm with text cross-attn | `detector_model.detr_encoder.*` |
| DETR Decoder | 6-layer post-norm, 200 queries, BoxRPB | `detector_model.detr_decoder.*` |
| Geometry Encoder | Box/point prompt encoding | `detector_model.geometry_encoder.*` |
| Mask Decoder | Pixel decoder + instance projection | `detector_model.mask_decoder.*` |
| Dot Product Scoring | Text-query classifier | `detector_model.dot_product_scoring.*` |
| **Multiplex Mask Decoder** | **16 objects simultaneously** | `tracker_model.sam_mask_decoder.*` |
| **Interactive SAM** | **Click/box prompt decoder** | `tracker_model.interactive_sam_*` |
| **Decoupled Memory Attention** | **4 layers with image cross-attn** | `tracker_model.memory_attention.*` |
| Memory Encoder | Multiplex-aware (32-channel mask input) | `tracker_model.memory_encoder.*` |

## File Structure

```
mlx_vlm/models/sam3_1/
├── __init__.py           # Module exports
├── config.py             # Configs (extends SAM 3 with multiplex_count, 3 scales)
├── vision.py             # TriViTDetNeck (imports ViT backbone from sam3)
├── sam_components.py     # MultiplexMaskDecoder, DecoupledMemoryAttention, SimpleRoPEAttention
├── tracker.py            # MultiplexTrackerModel (dual decoder, multiplex embeddings)
├── sam3_1.py             # Main Model + sanitize
├── processing_sam3_1.py  # Processor (same as SAM 3)
├── generate.py           # Inference pipeline (reuses SAM 3 generate.py)
└── convert_weights.py    # Meta .pt → MLX safetensors converter
```

### Code Reuse from SAM 3

SAM 3.1 imports these modules directly from `sam3/` (no duplication):
- ViT backbone, patch embeddings, windowed attention, 2D RoPE (`sam3.vision`)
- CLIP text encoder (`sam3.text_encoder`)
- DETR encoder + decoder (`sam3.encoder`, `sam3.decoder`)
- Position encodings (`sam3.position`)
- Geometry encoder (`sam3.geometry`)
- Detector segmentation head (`sam3.segmentation`)
- SAM prompt encoder, TwoWayTransformer (`sam3.sam_components`)

## Weight Conversion

SAM 3.1 weights are distributed as a Meta `.pt` checkpoint, not HuggingFace safetensors. The converter handles:
- Key remapping (`detector.*` → `detector_model.*`, `tracker.*` → `tracker_model.*`)
- QKV splitting (fused `in_proj_weight` → separate `q/k/v_proj`)
- CLS token stripping from position embeddings
- Conv2d/ConvTranspose2d transposition (PyTorch → MLX)
- Text projection transpose

```bash
python -m mlx_vlm.models.sam3_1.convert_weights --output /path/to/output
```

## License

The original SAM 3.1 model weights are released by Meta under the [**SAM License**](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE), a custom permissive license that grants a non-exclusive, worldwide, royalty-free license to use, reproduce, distribute, and modify the SAM Materials. Key points:

- Commercial and research use is permitted
- Derivative works must include a copy of the SAM License and attribution to Meta
- Provided "AS IS" without warranty
- Subject to applicable trade controls

By using it, you agree to the terms of Meta's SAM License. See the [full license text](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE) for details.
