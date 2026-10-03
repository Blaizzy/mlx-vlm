# K2-Horizon

K2-Horizon (`K2HorizonForCausalLM`) is IFM's family of dense, decoder-only text models: grouped-query attention, SwiGLU MLP, RMSNorm, an untied `lm_head`, and RoPE configured from the checkpoint's `rope_parameters` (YaRN on the 0.9B, plain RoPE on the larger variants). This port covers the dense text variants; the MoE / vision variants are out of scope.

## Supported Models

| Model ID | Notes |
|---|---|
| `IFM/K2-Horizon-0.9B` | Dense, YaRN RoPE, 64k vocab |
| `IFM/K2-Horizon-3.7B` | Dense |
| `IFM/K2-Horizon-7B` | Dense |
| `IFM/K2-Horizon-32B` | Dense |
| `IFM/K2-Horizon-MoVA-36B-A4B` | Not supported — MoE + MoVA vision tower |
| `IFM/K2-Horizon-375B-A23B` | Not supported — MoE |

## Model

| Variant | Hidden | Layers | Heads (Q/KV) | Head dim | Vocab | RoPE |
|---|--:|--:|--:|--:|--:|---|
| 0.9B | 1536 | 28 | 32 / 8 | 64 | 64256 | YaRN (θ 1e6) |
| 3.7B | 2560 | 36 | 32 / 8 | 128 | 250624 | plain (θ 1e7) |
| 7B | 4096 | 36 | 32 / 8 | 128 | 250624 | plain (θ 1e7) |
| 32B | 5120 | 64 | 64 / 8 | 128 | 250624 | plain (θ 1e7) |

The 3.7B / 7B / 32B share the 250624-token vocabulary; the 0.9B uses a distinct 64256-token tokenizer.

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model IFM/K2-Horizon-7B \
    --prompt "The capital of France is" \
    --max-tokens 64
```

With greedy decoding:

```bash
python -m mlx_vlm.generate \
    --model IFM/K2-Horizon-0.9B \
    --prompt "The capital of France is" \
    --max-tokens 64 \
    --temp 0.0
```
