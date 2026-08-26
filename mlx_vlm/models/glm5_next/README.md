# GLM-5-Next (`glm5_next`)

MLX support for the GLM-5-Next architecture, as shipped in **GLM-5.3-Flash**.

## Architecture

- **Hybrid decoder** — 34 Kimi-Delta linear-attention (KDA) layers interleaved with 11 DeepSeek sparse-attention (DSA) layers.
- **MLA with NoPE** and a **lightning indexer** (top-`index_topk` key selection over pooled keys).
- **288-expert MoE** (top-8) with a shared expert; **mHC hyper-connections**.
- A **multi-token-prediction (nextn) head** at the final layer, used for self-speculative decoding.

## Usage

```python
from mlx_vlm import generate, load

model, processor = load("zai-org/GLM-5.3-Flash")
print(generate(model, processor, "Explain multi-head latent attention.", max_tokens=256))
```

## Decode optimizations

All are on the compute path (no weight changes) and lossless:

| optimization | effect |
| --- | --- |
| KDA input-projection fusion | the six shared-input KDA projections become one (quantized) matmul via a lossless output-axis weight concat |
| Lightning-indexer chunked prefill | bounds prefill peak memory to `O(chunk · P)` (avoids a one-shot `O(S · P)` blow-up at long context) |
| Lightning-indexer incremental decode | per-step pool cost `O(T)` → `O(index_kpool)` (reuses stable complete pools) |
| Short-context dense-MLA bypass | when the cache fits within `index_topk` the indexer would select every token, so it is skipped and DSA falls through to dense MLA |
| Last-token `lm_head` | skip the vocab-wide projection on discarded prefill positions |
| FFN-block compile | `mx.compile` the stateless FFN half, scoped to single-stream decode |

## Self-speculative decoding (MTP)

GLM-5.3-Flash ships one trained nextn (MTP) layer inside the target checkpoint.
`mlx_vlm.convert` drops it during a normal conversion, so it is extracted into a
standalone drafter and used for self-speculative decoding. Each round the drafter
proposes one token from the target's hidden state and the target verifies the
`[bonus, draft]` block in a single forward.

- **Short-block verify gather.** The DSA verify path gathers each query's
  top-`index_topk` selected latents rather than masking over all keys, so
  verifying a short speculative block stays `O(index_topk)` instead of
  `O(context)`.
- **KDA block verify + rollback.** The linear-attention layers verify a block on
  the shared fused gated-delta kernel and roll the per-step recurrent state back
  to the accepted prefix on a partial-accept.
- **Never-lose adaptive pause.** With a single nextn head the block is 2 tokens,
  so a round only helps when the draft is accepted often enough to clear the
  verify overhead -- and that break-even shifts with context length and batch
  size. The orchestrator calibrates the drafting-vs-plain cost once, then runs a
  plain decode step whenever recent acceptance can't clear it (re-probing to
  resume). The target verifies every token, so the gate only trades drafting
  throughput and never drops below the baseline by more than the plain-step
  overhead.

See [`mlx_vlm/speculative/drafters/glm5_next_mtp/README.md`](../../speculative/drafters/glm5_next_mtp/README.md)
for the split tool and usage.

## Continuous batching

Runs the batched `BatchGenerator` path unmodified. The lightning indexer's incremental
pool and the DSA decode mask are batch-aware, so grow/shrink of the batch
(`filter`/`extend`) stays correct.
