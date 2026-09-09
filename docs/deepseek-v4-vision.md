# DeepSeek-V4 Flash Vision (experimental)

DeepSeek-V4 Flash Vision support covers the checkpoint's processor, vision
tower and aligner, image-span attention, vision-aware MoE routing, mixed
FP8/FP4 conversion, and its checkpoint-local DSpark drafter.

Convert the base checkpoint without materializing every source shard at once:

```bash
python -m mlx_vlm.models.deepseek_v4.convert \
  --hf-path deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --mlx-path DeepSeek-V4-Flash-Vision-Exp-MLX
```

Add `--mtp` to extract the three native DSpark stages beside the converted
target. The drafter must come from the same vision checkpoint; do not combine
it with the older DeepSeek-V4 Flash target or drafter.

The normal test suite uses small deterministic models to cover exact processor
layouts, all three compression ratios, text regression, portrait/landscape
images, multiple images, mixed text/image batches, repeated decode, feature
cache reuse, DSpark, and continuous-batching primitives.

Before advertising a converted checkpoint as supported, run the heavyweight
official-reference gates as well:

```bash
DEEPSEEK_V4_VISION_MLX_PATH=/path/to/converted-model \
DEEPSEEK_V4_VISION_FIXTURE=/path/to/reference-fixture.npz \
python -m pytest mlx_vlm/tests/test_deepseek_v4_reference.py -q
```

The `npz` fixture is exported from the official implementation for one fixed
RGB image and prompt. It contains `image_rgb`, `prompt`, `input_ids`,
`pixel_values`, every `image_*` processor array, `aligned_vision_features`, and
`first_token_logits`. Optional scalar `vision_atol`, `vision_rtol`,
`logits_atol`, and `logits_rtol` values override the default mixed-precision
tolerances.

To create it, check out the official model repository at the revision recorded
by the exporter, install `inference/requirements.txt`, and use the official
`inference/convert.py` to produce a single-rank runtime checkpoint (`MP=1`).
Then run:

```bash
python mlx_vlm/tests/export_deepseek_v4_reference.py \
  --official-inference-dir /path/to/DeepSeek-V4-Flash-Vision-Exp/inference \
  --checkpoint /path/to/DeepSeek-V4-Flash-Vision-Exp-TP1 \
  --image /path/to/fixed-image.png \
  --output /path/to/reference-fixture.npz
```

The exporter imports DeepSeek's own processor, vision tower, aligner, and model
from that checkout. It records the source revision, uses the repository's
actual image-placeholder token, and saves float32 views of BF16 intermediates so
NumPy can read the fixture without custom dtype support.
