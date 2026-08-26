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
  not actually faster on the current workload, re-probing periodically. Output is
  byte-identical either way (the target verifies every token), so this only puts
  a hard floor at the baseline decode speed.

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

## Notes

- The drafter is tied to the target family, size, and tokenizer it was split
  from; the runtime validates target/drafter hidden sizes match.
- The source model is licensed by its authors; mirror the license and
  attribution when hosting a split artifact.
