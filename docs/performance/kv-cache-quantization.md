# KV Cache Quantization

Reduce KV cache memory during continuous batching with `--kv-bits`. Both uniform quantization and TurboQuant are supported. Compatible with Automatic Prefix Caching (`APC_ENABLED=1`).

```sh
# Uniform 8-bit KV cache quantization
mlx_vlm.server --model google/gemma-4-26b-a4b-it --kv-bits 8

# TurboQuant 3.5-bit (3-bit keys + 4-bit values)
mlx_vlm.server --model google/gemma-4-26b-a4b-it --kv-bits 3.5 --kv-quant-scheme turboquant
```

Full-attention layers use quantized batch caches while sliding-window layers keep their fixed-size rotating caches. The last full-attention layer stays unquantized (sensitive in deep models).

## Per-tensor KV quantization

Keys and values do not have to share a bit-width or a backend. A fractional `--kv-bits` already splits the budget — `3.5` gives 3-bit keys and 4-bit values — and `--kv-key-bits` / `--kv-value-bits` override either side:

```sh
# 8-bit keys, 3-bit values, both TurboQuant
mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --kv-bits 3.5 --kv-quant-scheme turboquant \
  --kv-key-bits 8 --kv-value-bits 3
```

`--kv-key-scheme` / `--kv-value-scheme` go further and select a different backend per tensor, which builds a hybrid cache:

```sh
# uniform 8-bit keys beside TurboQuant 3-bit values
mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --kv-bits 8 --kv-quant-scheme uniform \
  --kv-value-bits 3 --kv-value-scheme turboquant
```

Two limitations apply to mixed *schemes* specifically:

- The hybrid cache dequantizes on every step instead of using a fused kernel, so it is slower than either homogeneous path.
- Mixed schemes are not supported during continuous batching or for batch prefix caches, and raise `NotImplementedError` there. Mixed bit-widths under a single scheme work everywhere.

Note that values are given the extra bit by default for a reason: value error passes straight through the attention output, whereas key error is partly reabsorbed by the softmax. Measured on Qwen3.5, spending an equal budget key-heavy was consistently worse than value-heavy, so prefer measuring before overriding.

Tested with gemma-4-26b-a4b-it at 20K context:

| Config | Gen tok/s | KV Cache | KV Reduction |
|--------|-----------|----------|--------------|
| No quant | 50.3 | 0.624 GB | 1x |
| Uniform 8-bit | 52.6 | 0.469 GB | **1.33x** |
| TurboQuant 3.5-bit | 25.6 | 0.365 GB | **1.71x** |

> Models with all full-attention layers (e.g. Qwen, LLaMA) see larger reductions — up to 3.6x at 8-bit and 6.4x at 4-bit.


## TurboQuant

TurboQuant compresses the KV cache during generation, enabling longer context lengths with less memory while maintaining quality.

### Quick Start

```sh
# 3.5-bit KV cache quantization (3-bit keys + 4-bit values)
mlx_vlm generate \
  --model mlx-community/Qwen3.5-4B-4bit \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant \
  --prompt "Your long prompt here..."
```

```python
from mlx_vlm import generate

result = generate(
    model, processor, prompt,
    kv_bits=3.5,
    kv_quant_scheme="turboquant",
    max_tokens=256,
)
```

```sh
# Server with TurboQuant
mlx_vlm server \
  --model google/gemma-4-26b-a4b-it \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

### How It Works

TurboQuant uses random rotation + codebook quantization ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) to compress KV cache entries from 16-bit to 2-4 bits per dimension:

- **Keys & Values**: MSE codebook quantization with Hadamard rotation
- **Fractional bits** (e.g. 3.5): uses lower bits for keys, higher for values (3-bit K + 4-bit V)

Custom Metal kernels fuse score computation and value aggregation directly on packed quantized data, avoiding full dequantization during decode.

### Performance

Tested on Qwen3.5-4B-4bit at 128k context:

| Metric | Baseline | TurboQuant 3.5-bit |
|--------|----------|-------------------|
| KV Memory | 4.1 GB | 0.97 GB (**76% reduction**) |
| Peak Memory | 18.3 GB | 17.3 GB (**-1.0 GB**) |

At 512k+ contexts, TurboQuant's per-layer attention is **faster than FP16 SDPA** due to reduced memory bandwidth requirements.

Tested on gemma-4-31b-it at 128k context:

| Metric | Baseline | TurboQuant 3.5-bit |
|--------|----------|-------------------|
| KV Memory | 13.3 GB | 4.9 GB (**63% reduction**) |
| Peak Memory | 75.2 GB | 65.8 GB (**-9.4 GB**) |

### Supported Bit Widths

| Bits | Compression | Best For |
|------|------------|----------|
| 2 | ~8x | Maximum compression, some quality loss |
| 3 | ~5x | Good balance of quality and compression |
| 3.5 | ~4.5x | Recommended default (3-bit keys + 4-bit values) |
| 4 | ~4x | Best quality, moderate compression |

### Compatibility

TurboQuant automatically quantizes `KVCache` layers (global attention). Models with `RotatingKVCache` (sliding window) or `ArraysCache` (MLA/absorbed keys) keep their native cache format for those layers since they are already memory-efficient.

TurboQuant is supported in both single-request generation and continuous batching on the server. In continuous batching mode, KV states are stored in TurboQuant's compressed format and dequantized at attention time (custom Metal kernels are not yet batch-aware).
