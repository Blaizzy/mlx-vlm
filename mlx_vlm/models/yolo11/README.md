# OmniParser v2.0 — MLX

OmniParser is a screen parsing tool that detects and captions UI elements using a two-stage pipeline: YOLO11 for icon detection + Florence-2 for captioning. This is the native MLX port.

## Model IDs

| Component | Model | Size |
|---|---|---|
| Icon Detector | [`axiom-of-choice/OmniParser-v2-icon-detect`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-detect) | 80 MB |
| Icon Captioner (bf16) | [`axiom-of-choice/OmniParser-v2-icon-caption-bf16`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-bf16) | 517 MB |
| Icon Captioner (8-bit) | [`axiom-of-choice/OmniParser-v2-icon-caption-8bit`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-8bit) | 356 MB |
| Icon Captioner (4-bit) | [`axiom-of-choice/OmniParser-v2-icon-caption-4bit`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-4bit) | 271 MB |

**Quantization Collection**: [axiom-of-choice/mlx-quantizations](https://huggingface.co/collections/axiom-of-choice/mlx-quantizations)

## Installation

```bash
pip install mlx-vlm
```

## Usage

### Python — Full Pipeline

```python
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.models.yolo11 import YOLO11, load_weights, non_max_suppression

# 1. Load detector
det_weights = mx.load("omniparser-icon-detect/model.safetensors")
detector = YOLO11(nc=1)
load_weights(detector, det_weights)

# 2. Detect icons
image = Image.open("screenshot.png").convert("RGB")
img_array = np.asarray(image).astype(np.float32) / 255.0
pred = detector(mx.array(img_array)[None])
mx.eval(pred)
detections = non_max_suppression(pred, conf_thresh=0.05, iou_thresh=0.1)[0]

# 3. Caption each detection
captioner, processor = load("axiom-of-choice/OmniParser-v2-icon-caption-8bit")
prompt = apply_chat_template(processor, captioner.config, "<CAPTION>", num_images=1)

results = []
for row in np.array(detections):
    x1, y1, x2, y2 = row[:4].astype(int)
    crop = image.crop((x1, y1, x2, y2)).resize((64, 64))
    out = generate(captioner, processor, prompt, image=crop, max_tokens=20, temperature=0.0)
    results.append({"bbox": row[:4].tolist(), "score": float(row[4]), "caption": out.text.strip()})

for r in results:
    print(f"{r['caption']} ({r['score']:.2f}) at {r['bbox']}")
```

### Detector Only

```python
import mlx.core as mx
from mlx_vlm.models.yolo11 import YOLO11, load_weights, non_max_suppression

model = YOLO11(nc=1)
load_weights(model, mx.load("model.safetensors"))

pred = model(mx.array(image)[None])  # (1, 5, N) — 4 box + 1 score
detections = non_max_suppression(pred, conf_thresh=0.05, iou_thresh=0.1)
```

## Architecture

```
Screenshot → YOLO11 (icon_detect) → Bounding boxes
                                        ↓
                              Crop each box → 64×64
                                        ↓
                              Florence-2 (icon_caption) → "Cancel", "Submit", ...
```

**Detector** (YOLO11-family):
- Backbone: C3k2 (CSP bottleneck with C3k inner blocks)
- Neck: SPPF + C2PSA (position-sensitive attention, 4 heads)
- Head: Anchor-free Detect with DFL box decoding, DW conv classification branch
- Classes: 1 (`icon`), trained at 1280px, inference at native resolution

**Captioner** (Florence-2-base fine-tune):
- Vision encoder + BART decoder
- Prompt: `<CAPTION>` (short description)
- Crops resized to 64×64 before captioning

## Quantization Fidelity

| Variant | Size | SNR | Cosine | Notes |
|---|---|---|---|---|
| bf16 | 517 MB | — | — | Reference |
| 8-bit g64 | 356 MB | 42.77 dB | 0.999974 | Recommended |
| 4-bit g64 | 271 MB | 20.84 dB | 0.995882 | Smaller, slight quality trade-off |

Vision tower stays bf16 in all quantized variants (mlx-vlm default). Effective bpw: 11.06 (8-bit), 8.40 (4-bit).

## File Structure

```
mlx_vlm/models/yolo11/
  __init__.py          # Public API
  yolo11.py            # Conv, C3k, C3k2, SPPF, C2PSA, Attention, Detect, YOLO11, NMS
  convert.py           # .pt → MLX safetensors converter
  README.md            # This file
```

## Reference

- [OmniParser](https://github.com/microsoft/OmniParser) — Microsoft's screen parsing tool
- [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) — Original weights
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) — Architecture reference
