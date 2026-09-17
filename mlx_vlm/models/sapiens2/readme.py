"""Task-specific README generator for the Sapiens2 MLX repos on HF Hub."""


_README_INTRO = """---
license: other
license_name: sapiens2-license
license_link: https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
library_name: mlx
tags:
  - mlx
  - sapiens2
  - vision
  - human-centric
  - {task}
pipeline_tag: image-to-image
base_model:
  - facebook/{hf_slug}
---

# mlx-community/{repo}

MLX port of [`facebook/{hf_slug}`](https://huggingface.co/facebook/{hf_slug}) at **{quant_human}** precision, converted with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

Sapiens2 is a family of human-centric ViTs pretrained on 1B human images.  This
repo contains the **{task}** head paired with the Sapiens2-{size} backbone.

## Install

```bash
pip install -U mlx-vlm
```
"""


_EXAMPLES = {
    "pose": '''## Usage — pose (308 keypoints)

The pose head is **top-down**: it expects the input image to be cropped to a
single person.  Three call modes are supported.

### Mode 1 — whole-image (single-person image, subject fills the frame)

```python
from pathlib import Path
from PIL import Image
import numpy as np
from mlx_vlm.utils import load_model
from mlx_vlm.models.sapiens2.processing_sapiens2 import Sapiens2Processor
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor

model = load_model(Path("mlx-community/{repo}"))
processor = Sapiens2Processor.from_pretrained("mlx-community/{repo}")
predictor = Sapiens2Predictor(model, processor)

result = predictor.predict(Image.open("person.jpg"))
# result.keypoints       (308, 2)  xy in original pixel coords (Sapiens keypoints-308 layout)
# result.keypoint_scores (308,)    heatmap peak confidences
best = np.argsort(result.keypoint_scores)[-10:]
for i in best:
    x, y = result.keypoints[i]
    print(f"kpt {{i:3d}}  ({{x:7.1f}}, {{y:7.1f}})  score={{result.keypoint_scores[i]:.3f}}")
```

### Mode 2 — top-down with **RTMDet** (paper-matched pipeline)

```python
import torch  # only for one-off state-dict conversion
from huggingface_hub import hf_hub_download
from mlx_vlm.models.rtmdet.config import RTMDetConfig
from mlx_vlm.models.rtmdet.rtmdet import Model as RTMDetModel
from mlx_vlm.models.rtmdet.processing_rtmdet import RTMDetProcessor
from mlx_vlm.models.rtmdet.generate import RTMDetPredictor
import mlx.core as mx

# Load the Sapiens RTMDet-m person detector (once)
rt = RTMDetModel(RTMDetConfig(arch="m"))
pth = hf_hub_download("facebook/sapiens-pose-bbox-detector",
                      "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
sd = torch.load(pth, map_location="cpu", weights_only=False)["state_dict"]
w = {{k: mx.array(v.detach().cpu().numpy())
     for k, v in sd.items() if "num_batches_tracked" not in k}}
rt.load_weights(list(RTMDetModel.sanitize(w).items()), strict=True)
rt.eval()
detector = RTMDetPredictor(rt, RTMDetProcessor(), score_threshold=0.3)

# Wire the detector into the pose predictor
predictor = Sapiens2Predictor(model, processor, detector=detector)
result = predictor.predict(Image.open("crowd.jpg"))
# result.persons: list of PersonPose — one per detected person
for p in result.persons:
    print(f"box={{tuple(int(v) for v in p.box)}}  det={{p.detector_score:.2f}}  "
          f"best_kpt_score={{p.keypoint_scores.max():.2f}}")
```

### Mode 3 — top-down with **RF-DETR** (reuses the rfdetr port in mlx-vlm)

```python
from mlx_vlm.utils import load_model as _load
from mlx_vlm.models.rfdetr.processing_rfdetr import RFDETRProcessor
from mlx_vlm.models.rfdetr.generate import RFDETRPredictor

rfdetr_model = _load(Path("mlx-community/rfdetr-base-fp32"))
rfdetr_proc  = RFDETRProcessor.from_pretrained("mlx-community/rfdetr-base-fp32")
rfdetr_det   = RFDETRPredictor(rfdetr_model, rfdetr_proc, score_threshold=0.3)

# RF-DETR returns all COCO classes; class index 1 is "person".
predictor = Sapiens2Predictor(model, processor,
                              detector=rfdetr_det, detector_class_filter=1)
result = predictor.predict(Image.open("crowd.jpg"))
```

### Mode 4 — top-down with **your own boxes**

```python
boxes = [(x1, y1, x2, y2), ...]  # from any detector, in original-image coords
result = predictor.predict(Image.open("crowd.jpg"), person_boxes=boxes)
```

All top-down modes expand each bbox to 3:4 aspect ratio with a 25% margin,
crop + resize to 1024×768, run pose, and stitch keypoints back to the
original-image coordinate system.  Output is the **308-keypoint** dense
whole-body skeleton (body, feet, face, hands) per person.
''',
    "seg": '''## Usage — body-part segmentation (29 classes)

```python
from pathlib import Path
from PIL import Image
import numpy as np
from mlx_vlm.utils import load_model
from mlx_vlm.models.sapiens2.processing_sapiens2 import Sapiens2Processor
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor

model = load_model(Path("mlx-community/{repo}"))
processor = Sapiens2Processor.from_pretrained("mlx-community/{repo}")
predictor = Sapiens2Predictor(model, processor)

result = predictor.predict(Image.open("person.jpg"))
# result.mask        (orig_h, orig_w) int32 class indices
# result.seg_logits  (29, H_out, W_out) raw logits

print("active classes:", np.unique(result.mask).tolist())
Image.fromarray(result.mask.astype(np.uint8)).save("mask.png")
```

Output: dense 29-class body-part segmentation (DOME 29-class scheme — face,
hair, torso, arms/legs split left/right, etc.).
''',
    "normal": '''## Usage — surface normals (dense XYZ)

```python
from pathlib import Path
from PIL import Image
import numpy as np
from mlx_vlm.utils import load_model
from mlx_vlm.models.sapiens2.processing_sapiens2 import Sapiens2Processor
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor

model = load_model(Path("mlx-community/{repo}"))
processor = Sapiens2Processor.from_pretrained("mlx-community/{repo}")
predictor = Sapiens2Predictor(model, processor)

result = predictor.predict(Image.open("person.jpg"))
normals = result.normal  # (3, H, W) unit-length XYZ vectors in camera space

rgb = np.clip((normals + 1) / 2, 0, 1).transpose(1, 2, 0)
Image.fromarray((rgb * 255).astype(np.uint8)).save("normals.png")
```

Output: per-pixel surface-normal vectors in 3-D camera frame, returned at the
original image resolution.
''',
    "pointmap": '''## Usage — pointmap (3-D XYZ + scale)

```python
from pathlib import Path
from PIL import Image
import numpy as np
from mlx_vlm.utils import load_model
from mlx_vlm.models.sapiens2.processing_sapiens2 import Sapiens2Processor
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor

model = load_model(Path("mlx-community/{repo}"))
processor = Sapiens2Processor.from_pretrained("mlx-community/{repo}")
predictor = Sapiens2Predictor(model, processor)

result = predictor.predict(Image.open("person.jpg"))
pm = result.pointmap  # (3, H, W) — X, Y, Z in canonical camera space
depth_z = pm[2]        # (H, W) — depth channel
scale = result.scale   # scalar: f_canonical / f_actual (focal-length ratio)

print(f"predicted scale = {{scale:.4f}}")
print(f"depth stats: min={{depth_z.min():.2f}}  max={{depth_z.max():.2f}}  "
      f"median={{np.median(depth_z):.2f}}")
```

Output: dense 3-D pointmap (X, Y, Z per pixel) plus a predicted focal-length
scale.  Multiply (depth, pointmap) by the scale to recover metric geometry.
''',
}


_README_TAIL = """## Convert your own checkpoint

```bash
# 1. Stage a float32 MLX directory from the Facebook checkpoint
python -m mlx_vlm.models.sapiens2.convert \\
    --hf-repo facebook/{hf_slug} \\
    --out ./{hf_slug}-fp32-mlx \\
    --dtype float32

# 2. Quantize + upload via the main mlx_vlm.convert CLI
python -m mlx_vlm.convert \\
    --hf-path  ./{hf_slug}-fp32-mlx \\
    --mlx-path ./{repo} \\
    {convert_flags} \\
    --upload-repo mlx-community/{repo}
```

## Architecture

Sapiens2 backbone: 2-D RoPE ViT (bf16 rope), partial GQA (full MHA in the
first/last 8 blocks, KV-half for the middle), SwiGLU FFN, cls + 8 storage
tokens.  Default input: **1024 × 768 (H × W)**, patch size 16, ImageNet
normalization on the [0, 255] scale.

See the [mlx-vlm sapiens2 port](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/sapiens2) for implementation details.

## License

Weights released under the [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md); this MLX repackaging inherits that license.

## Citation

```bibtex
@article{{khirodkarsapiens2,
  title  = {{Sapiens2}},
  author = {{Khirodkar, Rawal and Wen, He and Martinez, Julieta and Dong, Yuan
            and Su, Zhaoen and Saito, Shunsuke}},
  journal= {{arXiv preprint arXiv:2604.21681}},
  year   = {{2026}}
}}
```
"""


_QUANT_HUMAN = {
    "fp32": "float32 (no quant)",
    "bf16": "bf16 (cast only, no quant)",
    "4bit": "4-bit affine (group_size=64)",
    "5bit": "5-bit affine (group_size=64)",
    "6bit": "6-bit affine (group_size=64)",
    "8bit": "8-bit affine (group_size=64)",
    "mxfp4": "mxfp4 (microscaling fp4, group_size=32)",
    "mxfp8": "mxfp8 (microscaling fp8, group_size=32)",
    "nvfp4": "nvfp4 (NVIDIA fp4, group_size=16)",
}


_CONVERT_FLAGS = {
    "fp32": "--dtype float32",
    "bf16": "--dtype bfloat16",
    "4bit": "--quantize --q-bits 4 --q-group-size 64 --q-mode affine",
    "5bit": "--quantize --q-bits 5 --q-group-size 64 --q-mode affine",
    "6bit": "--quantize --q-bits 6 --q-group-size 64 --q-mode affine",
    "8bit": "--quantize --q-bits 8 --q-group-size 64 --q-mode affine",
    "mxfp4": "--quantize --q-bits 4 --q-group-size 32 --q-mode mxfp4",
    "mxfp8": "--quantize --q-bits 8 --q-group-size 32 --q-mode mxfp8",
    "nvfp4": "--quantize --q-bits 4 --q-group-size 16 --q-mode nvfp4",
}


def build_readme(task: str, size: str, quant: str) -> str:
    hf_slug = f"sapiens2-{task}-{size}"
    repo = f"{hf_slug}-{quant}"
    intro = _README_INTRO.format(
        task=task, size=size, repo=repo, hf_slug=hf_slug,
        quant_human=_QUANT_HUMAN[quant],
    )
    example = _EXAMPLES[task].format(repo=repo)
    tail = _README_TAIL.format(
        hf_slug=hf_slug, task=task, size=size, repo=repo,
        convert_flags=_CONVERT_FLAGS[quant],
    )
    return intro + "\n" + example + "\n" + tail


def main():
    """CLI: `python -m mlx_vlm.models.sapiens2.readme --task seg --size 0.4b --quant 4bit --out README.md`"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(_EXAMPLES.keys()))
    ap.add_argument("--size", required=True)
    ap.add_argument("--quant", required=True, choices=list(_QUANT_HUMAN.keys()))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).write_text(build_readme(args.task, args.size, args.quant))
    print(f"wrote README to {args.out}")


if __name__ == "__main__":
    main()
