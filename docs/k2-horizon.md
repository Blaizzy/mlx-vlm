# K2-Horizon

MLX port of the IFM/K2-Horizon family (`K2HorizonForCausalLM`) — a dense,
decoder-only text model: grouped-query attention, SwiGLU MLP, RMSNorm, an
untied `lm_head`, and RoPE configured from the checkpoint's `rope_parameters`
(plain RoPE on the ≥3.7B variants, YaRN on 0.9B).

## Variants

Covered by this implementation (dense text):

- K2-Horizon-0.9B (1.0B params, YaRN RoPE)
- K2-Horizon-3.7B (5B params)
- K2-Horizon-7B (9B params)
- K2-Horizon-32B (35B params)

Out of scope for this dense port:

- K2-Horizon-MoVA-36B-A4B (37B params) — MoE + MoVA vision tower
- K2-Horizon-375B-A23B (379B params) — MoE

## Usage

```bash
python -m mlx_vlm.generate --model IFM/K2-Horizon-7B \
    --prompt "The capital of France is" --max-tokens 64
```

Module: `mlx_vlm/models/k2_horizon/`.
