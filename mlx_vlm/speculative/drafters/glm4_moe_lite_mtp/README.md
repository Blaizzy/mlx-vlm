# GLM-4.7-Flash MTP Drafter

MLX support for the GLM-4.7-Flash (`glm4_moe_lite`) native Multi-Token
Prediction (MTP) drafter used by the speculative decoding path.

## What it is

GLM-4.7-Flash ships one trained nextn (MTP) layer inside the target checkpoint
at `model.layers.<num_hidden_layers>.*` — a dedicated `embed_tokens`, the
`enorm` / `hnorm` / `eh_proj` projections, an MLA `self_attn`, a 64-expert
MoE + shared expert, and an untied `shared_head`. The splitter writes those
weights into a standalone drafter folder with:

- `config.json` using `model_type: "glm4_moe_lite_mtp"`
- `model.safetensors` containing only the sanitized MTP weights, in the flat
  post-sanitize layout (absorbed-MLA `kv_b_proj` split into `embed_q` /
  `unembed_out`, experts stacked into `switch_mlp`)
- tokenizer files copied from the source model when present

The router correction bias (`mlp.gate.e_score_correction_bias`) is kept in
fp32 and the router gate weight is left full precision during quantization, as
GLM's `noaux_tc` routing requires.

## Split a Drafter

Run the splitter as a Python module:

```bash
# bf16 drafter
uv run python -m mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split \
  --model zai-org/GLM-4.7-Flash \
  --output ./GLM-4.7-Flash-MTP-bf16

# 4-bit drafter
uv run python -m mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split \
  --model zai-org/GLM-4.7-Flash \
  --output ./GLM-4.7-Flash-MTP-4bit \
  --q-bits 4 --q-group-size 64
```

Only the shards that hold the nextn tensors are read from the source repo.

Useful options:

- `--revision REV` to split from a specific Hugging Face revision.
- `--block-size N` to override the default speculative block size
  (`num_nextn_predict_layers + 1`).
- `--q-bits N` / `--q-group-size G` to affine-quantize the projection weights.
- `--force-download` to refresh the source model from Hugging Face.

Programmatic use:

```python
from mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split import (
    split_glm4_moe_lite_mtp,
)

split_glm4_moe_lite_mtp(
    source="zai-org/GLM-4.7-Flash",
    output="./GLM-4.7-Flash-MTP-bf16",
)
```

## Runtime

This drafter currently ships the split tool and the `glm4_moe_lite_mtp` config
only. The runtime drafter model class is deferred: mlx-vlm has no
`glm4_moe_lite` backbone in its model zoo yet, so it can neither load
GLM-4.7-Flash as a target nor build a sibling-style drafter (the
`qwen3_5_mtp` / `deepseek_v4_mtp` drafters are built on their in-zoo decoder
layers). The split output is a revision-pinned artifact consumed by external
MTP tooling; the runtime model class can drop in once the backbone lands.

## Notes

- The drafter is tied to the target family and tokenizer it was split from.
- The source model (`zai-org/GLM-4.7-Flash`) is MIT-licensed; mirror the
  license and attribution when hosting a split artifact.
