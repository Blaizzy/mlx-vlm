# DeepSeek-V4.1 DSpark drafter

`deepseek_v41_dspark` is the native speculative head shipped inside
`deepseek-ai/DeepSeek-V4.1-Flash`: 3 DSpark stages (`mtp.0..2.*`) drafting
`dspark_block_size` (5) tokens per step against target layers [37, 38, 39].

## Tensor layout (verified against Vontra's converted checkpoint)

- **Every stage**: MLA attention (`attn.wq_a/wq_b/wkv/wo_a/wo_b` + norms + sink),
  128 routed experts (`ffn.experts.N.w1/w2/w3`) + shared experts, gate
  (`ffn.gate.weight/bias/bias_vl`), Hyper-Connection tensors
  (`hc_attn/ffn_{fn,scale,base}`).
- **Stage 0 only**: `main_proj` + `main_norm` (target-hidden mixer).
- **Stage 2 only**: final `norm`, `markov_head` (embed + head) and
  `confidence_head.proj`.
- **Open question**: no `hc_head` tensors exist in the converted checkpoint
  while the reference `forward_head` applies one. The stage class must resolve
  whether the converter folded it (e.g. mean-collapse) before wiring the head.

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

The written config carries `n_mtp_layers`, `dspark_block_size`,
`dspark_target_layer_ids`, `dspark_noise_token_id` and `dspark_markov_rank`.

## Status

Config, `mtp.*` → `stages.*` split, and quantization profiles are implemented.
The `DSparkStage`/`DSparkAttention` draft loop (reusing the `markov_head` and
`confidence_head` classes from `mlx_vlm.models.deepseek_v41.dspark`, whose
`embed/head/proj` names already match this layout) is the remaining step.
