# anemll TBQ SWA

## Overview

This branch adds TurboQuant support for Sliding Window Attention (SWA) caches by converting `RotatingKVCache` into `TurboQuantRotatingKVCache` when `--turbo-kv-bits-swa` is provided.

### What was added

- `mlx_vlm/models/cache.py`
  - `TurboQuantRotatingKVCache`
  - `_python_quantize(...)` helper (single-token safety path)
  - conversion helper from `RotatingKVCache`
- `mlx_vlm/generate.py`
  - new CLI arg: `--turbo-kv-bits-swa`
  - quantization routing in `maybe_quantize_kv_cache(...)`

## SWA implementation notes

- SWA cache is kept in continuous temporal order while quantized key/value states are stored internally.
- **Critical Performance and Quality Detail:** It is absolutely critical *not* to de-quantize and re-quantize the entire SWA cache on every single generation step. In a naive implementation (like rolling a ring buffer or doing `to_quantized()` on the whole cache history), quantizing the full window of size N for every new token leads to two major issues:
  1. **Algorithmic Complexity:** It causes an $O(N^2)$ complexity for generation, drastically slowing down inference.
  2. **Quantization Degradation:** Repeatedly de-quantizing and re-quantizing the same tokens causes precision loss to compound over time, making the quantization quality fall apart and output turn into gibberish.
  
  Instead, `TurboQuantRotatingKVCache` strictly quantizes only the *new* tokens (O(1) per step) and concats/trims the underlying quantized state trees.
- Decode returns dense keys/values to attention while keeping compressed storage. This de-quantization is extremely fast compared to quantization.
- Global KV quantization still uses existing `--kv-bits` + `--kv-quant-scheme turboquant` flow.

## Usage example

Run from repo root (`anemll-mlx-vlm-swa`) with a VLM Gemma4 model:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-mlx-vlm-swa
source .venv/bin/activate
PYTHONPATH=. python -m mlx_vlm.generate \
  --model /Users/anemll/Models/gemma4/google-gemma-4-26B-A4B-it \
  --prompt "Make a game of Space Invaders in Python with modern visual effects" \
  --max-tokens 24000 \
  --temperature 0 \
  --kv-bits 4 \
  --kv-quant-scheme turboquant \
  --turbo-kv-bits-swa 3.5 \
  --quantized-kv-start 0
```

## Test examples

### 1) Baseline (no SWA TurboQuant)

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-mlx-vlm-swa
source .venv/bin/activate
PYTHONPATH=. python -m mlx_vlm.generate \
  --model /Users/anemll/Models/gemma4/google-gemma-4-26B-A4B-it \
  --prompt "Return Python hello world in one line." \
  --max-tokens 120 \
  --temperature 0
```

### 2) SWA TurboQuant enabled

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-mlx-vlm-swa
source .venv/bin/activate
PYTHONPATH=. python -m mlx_vlm.generate \
  --model /Users/anemll/Models/gemma4/google-gemma-4-26B-A4B-it \
  --prompt "Return Python hello world in one line." \
  --max-tokens 120 \
  --temperature 0 \
  --kv-bits 4 \
  --kv-quant-scheme turboquant \
  --turbo-kv-bits-swa 3.5 \
  --quantized-kv-start 0
```

### 3) Timed long-run test

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-mlx-vlm-swa
source .venv/bin/activate
/usr/bin/time -l PYTHONPATH=. python -m mlx_vlm.generate \
  --model /Users/anemll/Models/gemma4/google-gemma-4-26B-A4B-it \
  --prompt "Make a game of Space Invaders in Python with modern visual effects" \
  --max-tokens 24000 \
  --temperature 0 \
  --kv-bits 4 \
  --kv-quant-scheme turboquant \
  --turbo-kv-bits-swa 3.5 \
  --quantized-kv-start 0
```

## Quick validation command

```bash
PYTHONPATH=. uv run --active --python 3.11 --with-editable . python3 - <<'PY'
import importlib
import mlx.core as mx
from mlx_vlm.models import cache

mod = importlib.import_module('mlx_vlm.generate')
pc = [cache.RotatingKVCache(max_size=16, keep=0)]
mod.maybe_quantize_kv_cache(pc, 0, 64, 4, turbo_kv_bits_swa=3.5, kv_quant_scheme='turboquant')
assert type(pc[0]).__name__ == 'TurboQuantRotatingKVCache'
for _ in range(24):
    k = mx.random.normal((1,2,1,64)).astype(mx.float16)
    v = mx.random.normal((1,2,1,64)).astype(mx.float16)
    ok, ov = pc[0].update_and_fetch(k, v)
assert ok.shape[2] <= 16 and ov.shape[2] <= 16
print('SWA_CACHE_TEST: PASS')
PY
```

## Nightly Test Results

The overnight test suite finished with all 12000 tokens generated successfully. Here are the results:

```text
KL Divergence (4K tokens, full matrix)

Config             │ KL     │ TPS 
-------------------+--------+-----
FP16 Baseline      │ 0.0000 │ 86.5
SWA 4.0-bit        │ 0.0043 │ 41.1
SWA 3.5-bit        │ 0.0024 │ 42.5
SWA 3.0-bit        │ 0.0028 │ 43.9
Global 8-bit       │ 0.0002 │ 86.9
Global 4-bit       │ 0.0004 │ 87.0
SWA 4.0 + Global 4 │ 0.0023 │ 41.4
SWA 3.5 + Global 4 │ 0.0046 │ 42.5
SWA 3.0 + Global 4 │ 0.0034 │ 44.1

KL Divergence (12K tokens, long stream)

Config             │ KL     │ TPS 
-------------------+--------+-----
FP16 Baseline      │ 0.0000 │ 87.0
SWA 3.5-bit        │ 0.0084 │ 40.6
Global 4-bit       │ 0.0085 │ 85.2
SWA 3.5 + Global 4 │ 0.0085 │ 40.3

NIAH (All PASS at 8K & 12K context, depths 0.1/0.5/0.9)

Every configuration passed the needle retrieval test, including all depths for the 12K context with combined SWA 3.5 + Global 4.

Summary
•  Global 4-bit is the most mathematically faithful (lowest KL at both 4K and 12K).
•  SWA 3.5-bit adds roughly 0.008 KL at 12K tokens (still usable).
•  Combined compression (SWA 3.5 + Global 4) maintains passable KL (~0.008) while achieving ~2.6x memory reduction.
•  All NIAH tests passed — long-range context retrieval remains intact even at 12K tokens with aggressive quantization.
```

**Important Note on Testing Methodology:** 
Our initial nightly test runs used 3-bit quantized model weights (`unsloth-gemma-4-26b-a4b-it-UD-MLX-3bit`). 
For future rigorous evaluations of SWA KV Cache quantization degradation, it is **highly recommended** to run the tests using an **unquantized** model. Testing with unquantized weights ensures cleaner results by isolating the specific KV cache quantization effects from general model weight quantization noise.

## Future Improvements

- **Decoupled Key/Value Quantization:** Additional quality improvements can likely be achieved by adding K-specific quantization for SWA layers. For instance, keeping the Keys at 8-bit linear quantization or bypassing Key quantization entirely, while heavily quantizing Values to 4-bit or 3-bit. The underlying `mlx` TurboQuant Metal kernels (`_fused_mse_decode_kernel`) already support decoupled `key_bits` and `val_bits`; we just need to thread these options through the CLI and `maybe_quantize_kv_cache`.
