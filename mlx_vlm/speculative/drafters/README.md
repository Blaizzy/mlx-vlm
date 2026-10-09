# MTP drafters

Some base checkpoints ship native **multi-token-prediction (MTP)** tensors. This
package extracts them into a standalone MLX *drafter* for speculative decoding,
through one shared framework (`mtp_split.py`) with a small per-family subclass.

Registered native checkpoints with a drafter runtime can also be loaded
directly: pass the original checkpoint as `--draft-model` in the CLI or server.
The existing `MTP_SPLITTERS` registry selects the family, and the shared
`MTPSplitter.load()` prepares and loads its weights in memory without writing
a separate checkpoint. The draft kind is inferred from the splitter's output
model type, including native DSpark heads. See the
[Qwen MTP examples](qwen3_5_mtp/README.md).

## Extract a drafter

Standalone — auto-detects the family from the source config:

```bash
python -m mlx_vlm.split_mtp --model <hf-repo-or-local> --output <dir>
# --model-type <t>   force a specific splitter instead of auto-detect
# --q-bits N --q-group-size G   affine-quantize the drafter
```

Or fold it into conversion, so one command writes the base **and** the drafter:

```bash
python -m mlx_vlm.convert --hf-path <model> --mlx-path <out> --mtp
# --mtp-output <dir>   drafter location (default: <out>-mtp)
# with --quantize, the drafter is quantized to match the base (--q-bits/--q-group-size)
```

`--mtp` is opt-in and failure-isolated: if the source has no native MTP tensors,
or the split fails, the base conversion is unaffected. Because a normal MLX
conversion drops the MTP layer, the drafter is extracted from the original
source (which still carries `mtp.*`).

## Supported base model types

| base `model_type` | drafter `model_type` |
|---|---|
| `qwen3_5`, `qwen3_5_moe` | `qwen3_5_mtp` |
| `qwen3_next` | `qwen3_5_mtp` (reuses the runtime) |
| `qwen4_exp` | `qwen4_exp_mtp` |
| `deepseek_v4` | `deepseek_v4_mtp` |
| `glm4_moe_lite` | `glm4_moe_lite_mtp` |
| `inkling_mm_model` | `inkling_mtp` |

Detection is **tensor-presence based** — a config flag alone is not trusted
(some models declare MTP but ship no tensors; others ship tensors with no flag).

## Adding a family

1. Subclass `MTPSplitter` and override only what differs — `select_keys`,
   `rename` / `run_sanitize`, `postprocess`, `quantization_from_source`, `depth`,
   `extra_config`. The shared base handles shard discovery, loading, config
   assembly, in-memory model loading, tokenizer copy, and on-request affine
   quantization.
2. Register it by base `model_type` in `MTP_SPLITTERS`.

To *run* the drafter you also need a draft-model class. Reuse an existing one
when the architecture matches (e.g. Qwen3-Next reuses `qwen3_5_mtp`); otherwise
add a new one in its own `<family>_mtp/` directory.
