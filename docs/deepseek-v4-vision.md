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

The shared JSON model case checks language, vision, the aligner, text input
embeddings, and full-forward/cached-decode output shapes with a tiny model.
DSpark tests live in `mlx_vlm/tests/test_speculative.py`. These checks do not
establish numerical parity with the official implementation.
