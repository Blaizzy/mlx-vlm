# Qwen3.5 / Qwen3.6 / Qwen3.8 MTP Drafter

MLX support for Qwen3.5, Qwen3.6, and Qwen3.8 native Multi-Token Prediction
(MTP) drafters used by the speculative decoding path.

## What it is

These checkpoints can include native `mtp.*` weights in the target model. Those
weights are split into a standalone drafter folder with:

- `config.json` using `model_type: "qwen3_5_mtp"`
- `model.safetensors` containing only the sanitized MTP weights
- tokenizer files copied from the source model when present

At runtime, pass that folder as `--draft-model` with `--draft-kind mtp`.

## Split a Drafter

Run the splitter as a Python module:

```bash
uv run python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model Qwen/Qwen3.5-4B \
  --output ./Qwen3.5-4B-mtp
```

Useful options:

- `--revision REV` to split from a specific Hugging Face revision.
- `--block-size N` to override the default speculative block size.
- `--force-download` to refresh the source model from Hugging Face.

Programmatic use:

```python
from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp

split_qwen3_5_mtp(
    source="Qwen/Qwen3.5-4B",
    output="./Qwen3.5-4B-mtp",
)
```

## Optional compact proposal head

A Qwen MTP checkpoint does not contain an LM head. By default, the drafter uses
the target model's full LM head each time it proposes a token. For greedy
decoding, an optional compact head can reduce that work by projecting only a
selected subset of vocabulary rows.

The compact head only proposes tokens. The unchanged full target model still
verifies every proposal and decides which tokens are emitted. Sampled decoding
does not use the compact head.

### Build one

The builder copies packed affine rows directly from an existing quantized Qwen
LM head. It does not train, dequantize, or requantize them.

This command reproduces the selection used for the Qwen3.8 27B measurements
below. Pin the source revision so the result can be audited:

```bash
uv run python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.build_compact_head \
  --model orcarouter/Qwen3.8-27B-Uncensored-MLX \
  --revision b4603df5fd2a51e7fed2560ee7090caa4e13e4b7 \
  --subdir 4-bit \
  --vocab-prefix 98303 \
  --output ./Qwen3.8-27B-compact-head
```

With `--vocab-prefix`, tokenizer-added IDs are included by default and the row
count is padded with unused real IDs to a multiple of 32. This produced 98,336
rows for the checkpoint above. The selection is a model-specific performance
choice, not a recommended default for every language or workload. To use a
fully explicit selection instead, pass `--vocab-ids path/to/ids.json`, where
the file is a JSON array of real token IDs.

The output contains:

- `compact_head.safetensors`: packed `weight`, `scales`, affine `biases`, and
  the `vocab_ids` mapping;
- `config.json`: shape, quantization, target-model, tokenizer, source, and
  selection identity;
- `manifest.json`: hashes of the two required runtime files.

No model tensors are included in mlx-vlm. The person creating or distributing
a sidecar is responsible for the source model's license terms.

### Use it

```bash
uv run mlx_vlm.server \
  --model /path/to/Qwen3.8-27B-8bit \
  --draft-model /path/to/Qwen3.8-27B-mtp \
  --draft-kind mtp \
  --draft-compact-head ./Qwen3.8-27B-compact-head \
  --draft-block-size 4
```

For sidecars created by this builder, loading checks the model family, hidden
width, vocabulary size, tokenizer vocabulary, tensor shapes, and token-ID
mapping before attaching the compact head. Older sidecars without the `target`
metadata remain loadable, but only their shape and token range can be checked.
Rebuild them with this tool to enable the full checks.

### Performance measured on Qwen3.8 27B

A four-prompt, 1,024-token greedy screen on an M5 Max measured:

| Proposal projection | Pooled throughput | Change |
| --- | ---: | ---: |
| Full 8-bit head, 248,320 rows | 27.14 tok/s | baseline |
| Compact 8-bit head, 98,336 rows | 28.12 tok/s | +3.58% |
| Compact 4-bit head, 98,336 rows | 29.27 tok/s | +7.82% |

The compact 4-bit path was faster on all four prompts, added about 0.26 GiB of
peak MLX memory, and emitted the same 1,024 token IDs as the serial reference.
This was a directional screen on predecessor code, with one observation per
prompt and a production model resident. It is not a controlled result for this
PR head. A current-head compact-off/compact-on result should be reported before
using these percentages as the final performance claim.

The proposal-only compact-vocabulary approach and the 98,336-row physical shape
were informed by [MLX.fast challenge PR 59](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/pull/59).
The builder above uses the selected source checkpoint's own rows and tokenizer
mapping. It does not use challenge model tensors or its exact token mapping.

## Generate

```bash
uv run mlx_vlm.generate \
  --model Qwen/Qwen3.5-4B \
  --draft-model ./Qwen3.5-4B-mtp \
  --draft-kind mtp \
  --draft-block-size 4 \
  --prompt "Make a program to find pi" \
  --max-tokens 256 --temperature 0
```

## Server

```bash
uv run mlx_vlm.server \
  --model Qwen/Qwen3.5-4B \
  --draft-model ./Qwen3.5-4B-mtp \
  --draft-kind mtp \
  --draft-block-size 4
```

## Notes

- Greedy decoding (`temperature=0`) uses exact target verification.
- The drafter is tied to the target family and tokenizer it was split from.
- Batched Qwen MTP uses uniform acceptance to keep the drafter cache aligned.
- Official block-FP8 MTP weights are dequantized to BF16 while splitting.
- Multimodal prompts are supported, but image/video prefill still runs through
  the target model; MTP accelerates the text decode tail.
