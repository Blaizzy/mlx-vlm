# LongCat-Flash-Lite-Sparse (`longcat_flash_sparse`)

MLX support for **LongCat-Flash-Lite-Sparse** (`LongcatCausalLM`) — the LSA
sparse-attention + n-gram variant of the LongCat-Flash family. 

The Hugging Face config omits `model_type`; set `model_type: "longcat_flash_sparse"`
to dispatch here.

## Architecture

- **ScMoE decoder** — each of the 14 layers runs two attention blocks plus one
  MoE shortcut branch.
- **MLA** (multi-head latent attention) — q/kv-LoRA compression with absorbed
  `embed_q` / `unembed_out` projections.
- **LSA** (LongCat Sparse Attention) — a DeepSeek-style lightning indexer over
  MLA selects the top-`index_topk` (2048) keys, with streaming-aware indexing
  (fixed sink + local window) and cross-layer index reuse. Attention falls
  through to dense MLA while the cache is still within `index_topk`.
- **Zero-computation MoE** — 256 routed + 128 identity experts, top-12; an
  identity expert contributes its gate weight without a GEMM.
- **N-gram (`oe`) input embedding** — the token embedding is
  `word + Σ projections`, ~46% of the parameters.

## The n-gram fusion

The `oe` hash, tables, and projections match the published n-gram references.
The one difference in `LongcatCausalLM` is the **fusion**: it keeps the word
embedding at **full scale** — `word + Σ projections / (1 + num_embedders)` —
rather than the dense `longcat_flash_ngram` form
`(word + Σ projections) / (1 + num_embedders)`. Dividing the word by
`1 + num_embedders` garbles generation; this build applies the correct fusion.

## Usage

```python
from mlx_vlm import generate, load

model, processor = load(
    "AlazarM/LongCat-Flash-Lite-Sparse-4bit", trust_remote_code=True
)
print(generate(model, processor, "Explain multi-head latent attention.", max_tokens=256))
```

4-bit MLX checkpoint: <https://huggingface.co/AlazarM/LongCat-Flash-Lite-Sparse-4bit>

## Prefill optimizations

Both are on the compute path (no weight changes) and exact (≤1e-6 vs the
reference):

| optimization | effect |
| --- | --- |
| n-gram projection fusion | the 12 per-embedder projections become one GEMM over the concatenated lookups (12 launches → 1); ~1.5× on the projection |
| indexer epilogue kernel | replaces the ReLU + weighted head-sum epilogue (which materialized a `[B, 16, s, S]` tensor) with `deepseek_v4`'s fused Metal kernel; ~1.6× on the epilogue |
