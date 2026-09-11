# DeepSeek-V4.1 DSpark drafter

`deepseek_v41_dspark` is the native speculative head shipped inside
`deepseek-ai/DeepSeek-V4.1-Flash`: 3 DSpark stages (`mtp.0..2.*`) drafting
`dspark_block_size` (5) tokens per step against target layers [37, 38, 39].

It conforms to the shared `dflash` round loop (same contract as
`deepseek_v4_dspark`): cross-attention over the projected target hidden,
`VanillaMarkov` block sampling, `reset` / `_hidden` / `_logits` /
`draft_block`. Only the draft layers are V4.1-native (MLA cross-attention,
MoE, single-pass Hyper-Connections threaded internally from identity).

## Tensor layout (verified against Vontra's converted checkpoint)

- **Every stage**: MLA attention (`attn.wq_a/wq_b/wkv/wo_a/wo_b` + norms + sink),
  128 routed experts (`ffn.experts.N.w1/w2/w3`) + shared experts, gate
  (`ffn.gate.weight/bias/bias_vl`), Hyper-Connection tensors
  (`hc_attn/ffn_{fn,scale,base}`).
- **Stage 0 only**: `main_proj` + `main_norm` (target-hidden mixer).
- **Stage 2 only**: final `norm`, `markov_head` (embed + head, renamed onto
  `VanillaMarkov` as `markov_w1/markov_w2`) and `confidence_head.proj`
  (dropped: unused by the `dflash` loop).
- No `hc_head` exists: the reference collapses via `hc_pre` directly, which is
  what the port does.

## Building a drafter

```python
from mlx_vlm.speculative.drafters.deepseek_v41_dspark.split import (
    split_deepseek_v41_dspark,
)

split_deepseek_v41_dspark(
    "Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP",
    "DS-V41-DSpark-drafter",
)
```

The written config carries `n_mtp_layers`, `target_layer_ids`,
`mask_token_id`, `markov_rank` and `block_size`, so `load_drafter` resolves it
to `draft_kind="dflash"`.
