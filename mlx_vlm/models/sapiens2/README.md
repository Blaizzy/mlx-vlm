# Sapiens2 for MLX

Port of Meta's [Sapiens2](https://github.com/facebookresearch/sapiens2) — a
family of high-resolution ViTs pretrained on 1B human images — to mlx-vlm for
Apple Silicon.  Sapiens2 is **not a generative VLM**: each checkpoint pairs a
shared backbone with one of four dense-prediction heads (pose, seg, normal,
pointmap), so inference goes through `Sapiens2Predictor`, not
`mlx_vlm.generate`.

## Quick start

```python
from pathlib import Path
from PIL import Image
from mlx_vlm.utils import load_model
from mlx_vlm.models.sapiens2.processing_sapiens2 import Sapiens2Processor
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor

model = load_model(Path("./sapiens2-seg-0.4b-mlx"))
processor = Sapiens2Processor.from_pretrained("./sapiens2-seg-0.4b-mlx")
predictor = Sapiens2Predictor(model, processor)

result = predictor.predict(Image.open("person.jpg"))
# result.mask       (H, W) int class indices   (seg)
# result.keypoints  (308, 2) x/y pixel coords   (pose)
# result.normal     (3, H, W) unit vectors      (normal)
# result.pointmap   (3, H, W) XYZ  + .scale     (pointmap)
```

## Top-down pose (RTMDet or RF-DETR person detector)

The pose head is top-down — it expects the image to already be cropped to
a single person.  Pass an optional `detector=` to `Sapiens2Predictor` and
the predictor will run the detector, crop each person bbox to 3:4 with a 25%
margin, run pose, and stitch the keypoints back to full-image coordinates.
Two detectors are supported out of the box:

- **RTMDet** (Meta's Sapiens companion person detector) — see
  [`../rtmdet/README.md`](../rtmdet/README.md) for conversion + usage.
- **RF-DETR** (already ported in `mlx_vlm.models.rfdetr`) — pass
  `detector_class_filter=1` to keep only the COCO "person" class.

You can also pass `person_boxes=[(x1, y1, x2, y2), ...]` to `predict()` to
skip the detector entirely.  For the complete pose workflow with both
detector options (and a BYO-boxes variant), see
the per-task usage block in every `sapiens2-pose-*` quant README (generated
by `mlx_vlm.models.sapiens2.readme`).

## Converting a HF checkpoint

HF repos ship as bare `.safetensors` files — task + size are inferred from the
repo name:

```bash
python -m mlx_vlm.models.sapiens2.convert \
    --hf-repo facebook/sapiens2-seg-0.4b \
    --out ./sapiens2-seg-0.4b-mlx \
    --dtype bfloat16
```

This produces a self-contained MLX directory with `config.json`,
`preprocessor_config.json`, and sanitized `model.safetensors`.

## Available checkpoints

| Task       | 0.4 B                                  | 0.8 B                                  | 1 B                                  | 5 B                                  |
|------------|----------------------------------------|----------------------------------------|--------------------------------------|--------------------------------------|
| pose       | `facebook/sapiens2-pose-0.4b`          | `facebook/sapiens2-pose-0.8b`          | `facebook/sapiens2-pose-1b`          | `facebook/sapiens2-pose-5b`          |
| seg        | `facebook/sapiens2-seg-0.4b`           | `facebook/sapiens2-seg-0.8b`           | `facebook/sapiens2-seg-1b`           | `facebook/sapiens2-seg-5b`           |
| normal     | `facebook/sapiens2-normal-0.4b`        | `facebook/sapiens2-normal-0.8b`        | `facebook/sapiens2-normal-1b`        | `facebook/sapiens2-normal-5b`        |
| pointmap   | `facebook/sapiens2-pointmap-0.4b`      | `facebook/sapiens2-pointmap-0.8b`      | `facebook/sapiens2-pointmap-1b`      | `facebook/sapiens2-pointmap-5b`      |

Default head dimensions are wired for 0.4 B; larger sizes work but you may need
to override `HEAD_DEFAULTS` entries if the PT config diverges from the 0.4 B
shape (see `config.py`).

## Architecture

```
 (B, 1024, 768, 3) ─► PatchEmbed (16×16 Conv2d)
                    │
                    ▼
                 [cls | 8 storage | 64·48 patches]   ──► 2-D RoPE (bf16)
                    │
                    ▼
                 24/32/40/56 × TransformerBlock
                       ├─ RMSNorm + GQA (MHA outer 8 + inner GQA/2)
                       └─ RMSNorm + SwiGLU FFN (4× hidden)
                    │
                    ▼
                 RMSNorm → drop cls/storage → (B, 64, 48, C)
                    │
     ┌──────────────┼──────────────────┬──────────────────┐
     ▼              ▼                  ▼                  ▼
   Pose           Seg              Normal            Pointmap
  N×deconv ×2   K×deconv ×2   InputConv+4×PixShuf  +scale MLP
  +1×1 convs   +1×1 convs    +3×3 convs           +3×3 convs
  → 308 hm     → 29-class    → 3-ch unit normals  → 3-ch XYZ + s
```

## Closeness vs PyTorch (0.4 B, synthetic input)

Measured with a fixed-seed `np.random.standard_normal((1, 3, 1024, 768))` fp32
input against the reference [`facebookresearch/sapiens2`](https://github.com/facebookresearch/sapiens2):

| task     | max |Δ|    | mean |Δ|   | PT output std | agreement              |
|----------|-----------|------------|---------------|------------------------|
| seg      | 1.96e-01  | 4.77e-04   | 2.83          | 100.00% argmax match   |
| pose     | 5.05e-03  | 1.91e-05   | —             |  96.10% kpt-argmax     |
| normal   | ~1.0      | 3.92e-03   | 2.25          | — (dense regression)   |
| pointmap | 3.79e-01  | 4.00e-04   | —             | scale ∆ ≈ 1e-4         |

Drift is dominated by the backbone (max ≈ 2.5 e-3 on featmap output) and is
amplified by the heads' spatial upsamplers; argmax-based outputs are unaffected.

## Numerics notes

* **RoPE in bf16** — PT's `RopePositionEmbedding` defaults to
  `pos_embed_rope_dtype="bf16"`: `periods` is stored bf16, sin/cos are computed
  bf16, and `apply_rope` casts Q/K to bf16 around the rotation.  Running rope
  in fp32 inflates the backbone max-diff to ~5 e-2; matching PT's bf16 path
  drops it to ~2.5 e-3.
* **Conv weight layouts** — Conv2d is `(out, kH, kW, in)` as usual;
  `ConvTranspose2d` in the deconv heads needs `(in, out, kH, kW) → (out, kH,
  kW, in)`, which `Model.sanitize` handles via a key-hint match on
  `deconv_layers.`.
* **Channels-first Flatten in the pointmap scale branch** — PT's
  `nn.Sequential(Flatten, Linear, ...)` on a `(B, C, H, W)` tensor produces
  `[c0h0w0, c0h0w1, ...]` (channels outermost).  In MLX the conv output is
  `(B, H, W, C)`, so we `.transpose(0, 3, 1, 2)` before flattening to match the
  trained Linear layout.  Skipping this gives a wildly wrong scale.
* **PixelShuffle** — MLX has no builtin, so `heads.pixel_shuffle()` reshapes
  the channel axis as `(C, r_h, r_w)` to match PT's grouping, then transposes
  to interleave pixels.

## File map

```
sapiens2/
├── __init__.py                 module exports + framework aliases
├── config.py                   ModelConfig / BackboneConfig / HeadConfig
├── sapiens2.py                 top-level Model + sanitize()
├── vision.py                   Sapiens2Backbone (ViT + GQA + 2-D RoPE)
├── heads.py                    pose / seg / normal / pointmap heads
├── processing_sapiens2.py      Sapiens2Processor (1024×768, ImageNet norm)
├── generate.py                 Sapiens2Predictor + Sapiens2Result
├── convert.py                  HF → MLX converter (generates config.json)
├── language.py                 no-op LanguageModel stub for framework symmetry
└── README.md                   this file
```

## License

Weights and reference code are released under the
[Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md);
this MLX port inherits that license.

## Citation

```bibtex
@article{khirodkarsapiens2,
  title  = {Sapiens2},
  author = {Khirodkar, Rawal and Wen, He and Martinez, Julieta and Dong, Yuan
            and Su, Zhaoen and Saito, Shunsuke},
  journal= {arXiv preprint arXiv:2604.21681},
  year   = {2026}
}
```
