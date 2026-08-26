# GLM-5.3-Flash MTP Drafter

MLX support for the GLM-5-Next (`glm5_next`) native Multi-Token Prediction (MTP)
drafter used by the speculative-decoding path, as shipped in **GLM-5.3-Flash**.

## What it is

GLM-5.3-Flash ships one trained nextn (MTP) layer inside the target checkpoint
as a normal decoder layer at index `num_hidden_layers` (there is no top-level
`mtp.*` block). It has its own `enorm` / `hnorm` / `eh_proj` projections, a
DeepSeek sparse-attention `self_attn` (MLA with a lightning indexer), a
288-expert MoE + shared expert, and a `shared_head` norm. A normal MLX
conversion drops this layer, so the splitter extracts it into a standalone
drafter folder with:

- `config.json` using `model_type: "glm5_next_mtp"`
- `model.safetensors` containing only the sanitized nextn weights, in the flat
  post-sanitize layout (absorbed-MLA `kv_b_proj` split into `embed_q` /
  `unembed_out`, experts stacked into `switch_mlp`)
- tokenizer files copied from the source model when present

The router correction bias (`mlp.gate.e_score_correction_bias`) is kept in fp32
and the router gate weight is left full precision, matching the base model's
routing. If the source checkpoint is already quantized, the drafter records the
same quantization config.

The runtime drafter (`Glm5NextMTPDraftModel`) reuses the base model's
`Glm5NextMTP` block: each round it drafts one token from the target's hidden
state `h(t+1)` and the embedding of the accepted token `t+1`, the target
verifies the `[bonus, draft]` block in a single forward, and accepted tokens are
folded back into the drafter's own KV/indexer cache.

## How self-speculation works here

- **Block size 2.** With a single nextn layer the drafter proposes one token per
  round, so a round emits the always-free bonus token plus (when accepted) one
  drafted token.
- **Short-block verify gather.** The DSA verify path gathers each query's
  top-`index_topk` selected latents instead of masking over all keys, so
  verifying a 2-token block stays `O(topk)` rather than `O(context)`.
- **Never-lose adaptive pause.** A single nextn head only speeds things up when
  the draft is accepted often enough to clear the verify overhead, and that
  break-even shifts with context length and batch size. The orchestrator times
  drafting rounds against plain-decode rounds and pauses drafting whenever it is
  not actually faster on the current workload, re-probing periodically. The
  target verifies every token either way, so the gate only trades drafting
  throughput for a floor near the baseline decode speed.

## Split a Drafter

Run the splitter as a Python module:

```bash
uv run python -m mlx_vlm.speculative.drafters.glm5_next_mtp.split \
  --model zai-org/GLM-5.3-Flash \
  --output ./GLM-5.3-Flash-MTP
```

Only the shard(s) that hold the nextn tensors are read from the source repo. If
the source is a quantized MLX checkpoint, the extracted drafter inherits its
quantization.

Useful options:

- `--revision REV` to split from a specific Hugging Face revision.
- `--block-size N` to override the default speculative block size
  (`num_nextn_predict_layers + 1`).
- `--force-download` to refresh the source model from Hugging Face.

Programmatic use:

```python
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import split_glm5_next_mtp

split_glm5_next_mtp(
    source="zai-org/GLM-5.3-Flash",
    output="./GLM-5.3-Flash-MTP",
)
```

## Use it

```bash
mlx_vlm.generate --model ./GLM-5.3-Flash-mlx \
  --draft-model ./GLM-5.3-Flash-MTP --draft-kind mtp \
  --prompt "Explain multi-head latent attention." \
  --max-tokens 512 --temperature 0
```

## Measured speedups

M3 Ultra, GLM-5.3-Flash 4-bit, greedy decoding, decode tok/s with vs. without the
drafter (same build), 128-token completions:

| batch | context | no MTP | MTP | speedup | accept |
| --: | --: | --: | --: | --: | --: |
| 1 | 512   | 33.8 | 30.8 | 0.91x | 0.65 |
| 1 | 2048  | 29.8 | 34.6 | 1.16x | 0.79 |
| 1 | 8192  | 29.3 | 35.7 | 1.22x | 0.87 |
| 1 | 32768 | 28.6 | 28.1 | 0.98x | 0.75 |
| 2 | 512   | 51.9 | 47.9 | 0.92x | 0.78 |
| 2 | 2048  | 45.5 | 52.5 | 1.16x | 0.81 |
| 2 | 8192  | 45.1 | 54.1 | 1.20x | 0.89 |
| 2 | 32768 | 43.1 | 41.1 | 0.95x | 0.74 |
| 4 | 512   | 82.3 | 73.7 | 0.90x | 0.56 |
| 4 | 2048  | 71.7 | 81.2 | 1.13x | 0.79 |
| 4 | 8192  | 70.5 | 78.9 | 1.12x | 0.81 |
| 4 | 32768 | 66.3 | 60.9 | 0.92x | 0.56 |

The win is in the mid-context band (1.12-1.22x at 2k-8k across batch sizes). A
single nextn head drafts one token per round, so the block is 2 and the ceiling
is `(1 + accept) / verify_ratio` -- with `verify_ratio` ~= 1.4 that is about
1.36x at realistic acceptance and ~1.45x even at perfect acceptance. Larger gains
need a multi-layer MTP head. At very short and very long context (and as the batch
grows) drafting can't clear the verify overhead; the never-lose gate detects this
and falls back to plain decode, so those cells stay within the plain-step overhead
of the baseline instead of regressing.

## Notes

- The drafter is tied to the target family, size, and tokenizer it was split
  from; the runtime validates target/drafter hidden sizes match.
- The source model is licensed by its authors; mirror the license and
  attribution when hosting a split artifact.
