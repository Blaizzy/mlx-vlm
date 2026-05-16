# OWLv2 (Open-World Localization v2) for MLX

MLX port of [Google's OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) — an open-vocabulary object detection model (~200M parameters).

- **Paper:** [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) (Minderer et al., 2023)

## Quick Start

```python
import mlx.core as mx
from PIL import Image
from mlx_vlm.models.owlv2.config import ModelConfig
from mlx_vlm.models.owlv2.owlv2 import Model
from mlx_vlm.models.owlv2.processing_owlv2 import OWLv2Processor
from mlx_vlm.models.owlv2.generate import postprocess

# Load model
from safetensors import safe_open
from huggingface_hub import hf_hub_download

path = hf_hub_download("google/owlv2-base-patch16-ensemble", "model.safetensors")
weights = {}
with safe_open(path, framework="numpy") as f:
    for k in f.keys():
        weights[k] = mx.array(f.get_tensor(k))

model = Model(ModelConfig())
model.load_weights(list(Model.sanitize(weights).items()))
processor = OWLv2Processor(image_size=960)

# Detect
image = Image.open("photo.jpg")
inputs = processor.preprocess_image(image)
query_labels = ["a cat", "a dog"]
input_ids, attention_mask = processor.tokenizer.tokenize(query_labels)

outputs = model(
    mx.array(inputs["pixel_values"]),
    mx.array(input_ids),
    mx.array(attention_mask),
)
mx.eval(outputs["pred_logits"], outputs["pred_boxes"], outputs["objectness_logits"])

result = postprocess(outputs, inputs["original_size"], query_labels, score_threshold=0.1)
for name, score, box in zip(result.class_names, result.scores, result.boxes):
    print(f"{name}: {score:.2f} [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

## Model Variants

| Model | Params | Image Size | Patch Size |
|-------|--------|------------|------------|
| `google/owlv2-base-patch16` | 200M | 768 | 16 |
| `google/owlv2-base-patch16-ensemble` | 200M | 960 | 16 |
| `google/owlv2-large-patch14` | 900M | 840 | 14 |
| `google/owlv2-large-patch14-ensemble` | 900M | 1008 | 14 |

## Architecture

OWLv2 uses a CLIP backbone (ViT vision encoder + text encoder) with three detection heads:

- **Class head**: cosine similarity between image patch embeddings and text query embeddings, with learned per-token shift/scale
- **Box head**: MLP predicting cxcywh offsets from grid-biased positions
- **Objectness head**: query-agnostic binary classifier for filtering background patches
