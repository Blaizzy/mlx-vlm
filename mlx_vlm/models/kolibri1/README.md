# Kolibri 1 (`kolibri1`)

MLX support for Aleph Alpha's Kolibri 1, a German/English text model for
reasoning and tool calling.

## Models

| Hugging Face model | Weights |
|---|---|
| `Aleph-Alpha/Kolibri-1` | block FP8 (loads as MXFP8) |
| `Aleph-Alpha/Kolibri-1-BF16` | BF16 |

The router, embeddings and `lm_head` stay in BF16, as in the checkpoint.

## Architecture

- 78B total, 3.5B active parameters, 50 decoder layers.
- MoE in every layer: 384 routed experts (top 6) and one ungated shared expert.
- Routing selects the top-k on `logits + expert_bias`. Weights are
  `sigmoid(logits)` of the selected experts, without renormalisation.
- GQA (48 query heads, 4 KV heads) with per-head QK RMSNorm.
- Sliding-window and full attention in a 4:1 ratio. Only the sliding-window
  layers use RoPE.
- Sandwich norms after attention and after the MoE.

## Usage

```sh
mlx_vlm.generate --model Aleph-Alpha/Kolibri-1 \
  --prompt "Erkläre Mixture-of-Experts in drei Sätzen." --max-tokens 512
```

```python
from mlx_vlm import generate, load

model, processor = load("Aleph-Alpha/Kolibri-1")
print(generate(model, processor, "Explain mixture-of-experts.", max_tokens=512))
```

## Tool calling

The chat template uses Hermes-style `<tool_call>` JSON. The server selects the
existing `json_tools` parser from the template.
