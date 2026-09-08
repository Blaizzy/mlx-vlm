# Spark-X2.5

Spark-X2.5 (`Spark2_5ForCausalLM`) is a dense, decoder-only text model with a
hybrid attention stack. Each block uses a fused `q_k_v` projection, grouped-query
attention (16 Q / 4 KV heads, head dim 256), and a per-head sigmoid gate applied
to the attention output. Layers alternate between **sliding-window attention**
(window 512) and **full attention** in a `[S, S, S, F]` pattern (27 sliding, 9
full), each layer type carrying its own partial-rotary RoPE. The MLP is GeGLU,
normalization is RMSNorm, and the `lm_head` is tied to the token embedding.

## Supported Models

| Model ID | Notes |
|---|---|
| `XHToken/Spark-X2.5-4B` | 4B instruct |
| `XHToken/Spark-X2.5-4B-Base` | 4B base pretrain |
| `XHToken/Spark-X2.5-1.7B` | 1.7B instruct |
| `XHToken/Spark-X2.5-1.7B-Base` | 1.7B base pretrain |

All four share the `spark2_5` architecture, tied embeddings, a 128k vocab, and a
512-token sliding window; the two sizes differ only in width/depth.

## Model

| Size | Hidden | Layers (S / F) | Heads (Q / KV) | Head dim | Vocab | Window |
|---|--:|:--:|--:|--:|--:|--:|
| 4B | 2560 | 36 (27 / 9) | 16 / 4 | 256 | 131072 | 512 |
| 1.7B | 2048 | 28 (21 / 7) | 8 / 2 | 256 | 131072 | 512 |

Both use the same RoPE layout: full-attention layers apply a partial rotary
factor of 0.25 (64 of 256 dims) with `θ = 5e6`, while sliding layers rotate all
256 dims with `θ = 1e4`. Sliding layers are served from a rotating KV cache
bounded to the 512-token window, so decode throughput and memory stay flat as
context grows.

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model XHToken/Spark-X2.5-4B \
    --prompt "The capital of France is" \
    --max-tokens 64
```

With greedy decoding:

```bash
python -m mlx_vlm.generate \
    --model XHToken/Spark-X2.5-4B \
    --prompt "The capital of France is" \
    --max-tokens 64 \
    --temp 0.0
```
