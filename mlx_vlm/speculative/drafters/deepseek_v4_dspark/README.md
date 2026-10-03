# DeepSeek-V4 DSpark drafter

`deepseek_v4_dspark` is the native DSpark speculative head shipped inside
DeepSeek-V4 checkpoints including `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.
It is a **DeepSeek-V4-backbone variant of
the model-agnostic `dspark` drafter**: it reuses the shared DSpark machinery —
`VanillaMarkov` block sampling and the `dflash` round loop with its target-hidden
tap — and only swaps the Qwen-style draft layers for DeepSeek-V4 blocks (MLA
attention, MoE, Hyper-Connections). It is distinct from `deepseek_v4_mtp`, which
serves the single-block MTP of the base `deepseek-ai/DeepSeek-V4-Flash`.

## Architecture

Like the base `dspark` drafter, it conforms to the DFlash drafter contract
(`reset` / `_hidden` / `_logits` / `draft_block`) and maps to `draft_kind="dflash"`:

- A stack of `n_mtp_layers` DeepSeek-V4 HC transformer stages. **Stage 0** owns
  `main_proj` (`hidden_size * len(target_layer_ids) -> hidden_size`) + `main_norm`,
  which mix the concatenated hidden states of the target's `target_layer_ids`
  (the `dflash` capture tap, mean-pooled over the hyper-connection copies) into
  the attention context. **The last stage** owns `hc_head` + `norm`.
- The block attention reads that projected context (cached) and denoises a block
  of `block_size - 1` proposals seeded with `mask_token_id`.
- A model-level `markov_head` (`VanillaMarkov`, reused from `..dspark`) applies a
  low-rank bigram bias while sampling the block.

The confidence head shipped in the checkpoint is unused by the `dflash` loop and
is dropped during `sanitize`.

## Building a drafter

```python
from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (
    split_deepseek_v4_dspark,
)

split_deepseek_v4_dspark(
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "DS-V4-Vision-DSpark-drafter",
)
```

Always extract the drafter from the same source checkpoint as the target. The
vision checkpoint's DSpark and target weights belong to its continued-trained
language model and must not be mixed with the older V4 Flash drafter.

The written config carries `n_mtp_layers`, `block_size`, `target_layer_ids`,
`mask_token_id` and `markov_rank`, so `load_drafter` resolves it to
`draft_kind="dflash"`.
