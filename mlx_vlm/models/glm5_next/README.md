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

## Image input

`Glm5NextProcessor` is registered for `model_type=glm5_next`. Two details that
are easy to get wrong if you reuse GLM-OCR helpers:

1. **Token-budget `smart_resize`.** GLM-5.3's `min_image_tokens` / `max_image_tokens`
   are token counts. They are converted to pixels with
   `temporal_patch_size * (patch_size * merge_size)²`. The GLM-OCR / Qwen helper
   treats the same numbers as a raw pixel budget and will shrink a native 448²
   tile.
2. **One-shot `<|image|>` expansion.** Replacement is `image_token * N`. Split on
   the original slots before inserting the run. A `while token in text` loop
   reconsumes the expansion and asks for extra `image_grid_thw` rows.

Image markers come from the checkpoint's own `chat_template.jinja`: upstream
revision `690b7052` (2026-09-04) and later emit
`<|begin_of_image|><|image|><|end_of_image|>` for image parts. Older snapshots of
the template rendered images as an "unable to process" reminder, so re-download
`chat_template.jinja` if your local copy predates that revision.

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("zai-org/GLM-5.3-Flash")
prompt = apply_chat_template(
    processor, model.config, "Describe this image.", num_images=1
)
print(generate(model, processor, prompt, image=["photo.jpg"], max_tokens=256))
```

Video preprocessing is not implemented yet.

A 448² native tile is 256 LLM image tokens (`32×32` patches, `2×2` merge).

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

## Continuous batching

Runs the batched `BatchGenerator` path unmodified. The lightning indexer's incremental
pool and the DSA decode mask are batch-aware, so grow/shrink of the batch
(`filter`/`extend`) stays correct.
