# Z1T-0

Z1T-0 is Extropic's open-weight, **attention-free** decoder LM, ported from JAX/Equinox
([extropic-ai/sparse-transformers](https://github.com/extropic-ai/sparse-transformers),
[writeup](https://extropic.ai/writing/z1t)). It has no softmax attention: each block mixes
tokens with AFT-conv — a causal depthwise conv plus a causal cumulative pool. Dynamic-Tanh
(DyT) is the only normalization, and every internal projection is a fixed fan-in-`k` sparse
`tanh`-linear. Decoding is O(1) per token via a small per-layer streaming cache (running pool
sums + conv window). Tokenizer is GPT-2 BPE.

## Supported Models

| Model ID | Notes |
|---|---|
| `AlazarM/Z1T-0-mlx` | MLX conversion of the open weights (fp32) |
| `Extropic-AI/Z1T-0` | Original JAX/Equinox `.eqx` checkpoint — convert to MLX first |

## Model

| Vocab | Hidden | Layers | AFT heads | Conv ksize | Fan-in | Max pos |
|--:|--:|--:|--:|--:|--:|--:|
| 50257 | 12288 | 4 | 4 | 4 | 4 | 256 |

The published checkpoint ships a numerically-zero positional table (AFT-conv already carries
local position), so absolute position is effectively unused.

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model AlazarM/Z1T-0-mlx \
    --prompt "The meaning of life is" \
    --max-tokens 64 \
    --temp 0.0
```

This is a small research model (4 layers); generations are short and repetitive by design.
