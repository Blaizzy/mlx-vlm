# RT-DETRv2 for MLX

MLX port of [RT-DETRv2](https://arxiv.org/abs/2407.17140), the real-time object detection transformer. Loads any HuggingFace `RTDetrV2ForObjectDetection` checkpoint (ResNet-50-vd or ResNet-101-vd backbone) on Apple Silicon.

Validated against `transformers.RTDetrV2ForObjectDetection` end-to-end on both backbones: max abs error ~2e-5 on logits, sub-pixel on bboxes for real document inputs.

## Use

Pre-converted bf16 checkpoints are on the Hub:

- [`mlx-community/docling-layout-heron-mlx-bf16`](https://huggingface.co/mlx-community/docling-layout-heron-mlx-bf16) — ResNet-50-vd backbone, 86 MB
- [`mlx-community/docling-layout-heron-101-mlx-bf16`](https://huggingface.co/mlx-community/docling-layout-heron-101-mlx-bf16) — ResNet-101-vd backbone, 154 MB

```python
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from mlx_vlm.utils import load_model
from mlx_vlm.models.rt_detr_v2.generate import RTDetrV2Predictor
import mlx_vlm.models.rt_detr_v2  # registers the processor with AutoProcessor

path = Path(snapshot_download("mlx-community/docling-layout-heron-mlx-bf16"))
model = load_model(path)
processor = AutoProcessor.from_pretrained(path)
predictor = RTDetrV2Predictor(model, processor, threshold=0.3)

result = predictor.predict(Image.open("page.png"))
for name, score, box in zip(result.class_names, result.scores, result.boxes):
    print(f"{name:20s} {score:.3f} {box.tolist()}")
```

## Convert your own

```bash
python -m mlx_vlm.models.rt_detr_v2.convert \
    --hf-path docling-project/docling-layout-heron \
    --output ./docling-layout-heron-mlx-bf16 \
    --dtype bfloat16
```

The output directory contains `model.safetensors`, `config.json`, and `preprocessor_config.json`. The converter runs a forward pass via `mlx_vlm.utils.load_model` before returning, so a successful run means the checkpoint loads cleanly.

`--dtype` accepts `float16`, `bfloat16`, or `float32` (the canonical set in `mlx_vlm.utils.MODEL_CONVERSION_DTYPES`).

`result` is a `DetectionResult` with vectorized fields:

```
result.boxes        # (N, 4) xyxy in original-image pixels
result.scores       # (N,) confidence in [0, 1]
result.labels       # (N,) integer class ids
result.class_names  # list of str resolved via config.id2label
```

The processor is auto-registered with `transformers.AutoProcessor` at import time (see `install_auto_processor_patch` in `mlx_vlm/models/base.py`), so `AutoProcessor.from_pretrained` dispatches into `RTDetrV2Processor` for any directory whose `config.json` has `model_type: rt_detr_v2`.

## Architecture

```
Image (HxW NHWC) -> ResNet-50/101-vd backbone (strides 8/16/32)
                 -> EncoderInputProj (1x1 conv + BN per level)
                 -> HybridEncoder: AIFI (deepest level) + FPN + PAN
                 -> Encoder query selection (top-K on flat positions x labels)
                 -> Deformable-attention decoder (6 layers, iterative bbox refinement)
                 -> {pred_logits (B, Q, num_labels), pred_boxes (B, Q, 4)}
```

The backbone depth (`[3, 4, 6, 3]` for ResNet-50, `[3, 4, 23, 3]` for ResNet-101) is read from `backbone_config.depths` so the same module handles both variants.

## File structure

```
mlx_vlm/models/rt_detr_v2/
  config.py                  # Dataclass configs (top-level + backbone/encoder/decoder)
  vision.py                  # ResNet-vd backbone + hybrid encoder (AIFI + FPN + PAN)
  transformer.py             # MSDeformableAttention + decoder + anchor priors
  rt_detr_v2.py              # Model wrapper + Model.sanitize
  generate.py                # RTDetrV2Predictor + DetectionResult
  processing_rt_detr_v2.py   # Image preprocessing + AutoProcessor registration
  convert.py                 # HF -> MLX checkpoint converter
  language.py                # Stub (framework compatibility)
```

Trainability is preserved on the MLX side: BatchNorms remain as layers (not folded into preceding convs), `denoising_class_embed` is loaded as a real parameter, and `n_points_scale` in `MSDeformableAttention` is a live buffer rather than a hard-coded constant.

## References

- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)
- [HuggingFace `transformers` RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)
- [Docling layout models on HF](https://huggingface.co/docling-project) (ResNet-50: `docling-layout-heron`, ResNet-101: `docling-layout-heron-101`)
