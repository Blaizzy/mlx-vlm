# Hy4 MTP Drafter

Hy4 stores one native next-token prediction layer at `model.mtp_layers.0.*`.
The splitter extracts that layer into a standalone `hy_v4_mtp` drafter while
the target keeps ownership of its embedding table, LM head, and KV cache.

## Split

```bash
python -m mlx_vlm.split_mtp \
  --model tencent/Hy4-preview-FP8 \
  --output ./Hy4-preview-MTP
```

The source MTP block has rank-3 hidden states and fuses `enorm(token_embed)`
with `hnorm(target_hidden)` through `eh_proj`; it is not a hyper-connection
head. The emitted configuration defaults to a two-token verification block:
one native draft token plus one target bonus token.

## Generate

```bash
mlx_vlm.generate \
  --model /path/to/Hy4-preview-4bit \
  --draft-model /path/to/Hy4-preview-MTP-4bit \
  --draft-kind mtp \
  --max-tokens 256 \
  --temperature 0
```

For quantized Hy4 targets, the verifier keeps the absorbed-MLA representation
used by one-token decode across the whole verification block. This preserves
decode-equivalent algebra while still verifying the block in one target pass.
