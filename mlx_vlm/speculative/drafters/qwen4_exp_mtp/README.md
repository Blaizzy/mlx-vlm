# Qwen4-Exp MTP

This drafter runs the native multi-token prediction head embedded in Qwen4-Exp
checkpoints. Extract it with the generic splitter, then pass the resulting
folder as an MTP draft model:

```bash
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP

mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --draft-model ./Qwen3.8-Flash-Next-MTP \
  --draft-kind mtp \
  --max-tokens 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

The native checkpoint contains one MTP layer, so the default draft block is
one speculative token. A larger `--draft-block-size` chains the same head
autoregressively; whether that is faster depends on prompt and hardware.

## Optional quantization

The head can be quantized independently when it is extracted:

```bash
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP-3bit \
  --q-bits 3 \
  --q-group-size 32
```

A locally converted target model can be supplied to `--model` in the
generation command as well.
