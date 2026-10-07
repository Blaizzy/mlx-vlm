# Clef decisions on MLX

Prepared checkpoints of [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
(27B) and [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B):

| Model | 8-bit | NVFP4 | MXFP4 |
|---|---|---|---|
| Clef | [nativ-community/clef-MLX-8bit](https://huggingface.co/nativ-community/clef-MLX-8bit) | [nativ-community/clef-MLX-NVFP4](https://huggingface.co/nativ-community/clef-MLX-NVFP4) | [nativ-community/clef-MLX-MXFP4](https://huggingface.co/nativ-community/clef-MLX-MXFP4) |
| Clef-Flash | [nativ-community/clef-flash-MLX-8bit](https://huggingface.co/nativ-community/clef-flash-MLX-8bit) | [nativ-community/clef-flash-MLX-NVFP4](https://huggingface.co/nativ-community/clef-flash-MLX-NVFP4) | [nativ-community/clef-flash-MLX-MXFP4](https://huggingface.co/nativ-community/clef-flash-MLX-MXFP4) |

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/clef-flash-MLX-8bit")
result = predict(model, processor, "Our checkout is returning errors", {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "bool", "instructions": "Is a service down?"},
})
print(result["answers"]["department"]["value"])
```

To prepare another checkpoint, set `model_type: "clef"` in its root
`config.json`, add a `head_config` object holding the published
`joint_head_config.json`, and list the `joint_head.safetensors` tensors in
`model.safetensors.index.json` alongside the backbone shards. Weights use the
existing Qwen3.5 sanitization; `mlx_vlm.convert` keeps the head unquantized.

Clef supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. All
questions in one request are scored jointly in a single forward pass, and no
answer tokens are generated. Instructions are optional and default to the
question name. Choice criteria are labels or a mapping of labels to
descriptions, score criteria are an ordered list of level descriptions, and
Boolean criteria optionally describe `true` and `false`.

Each answer has `type` and `value`. Choice and score answers include
`probabilities` and a `confidence` in `metadata`, the probability of the chosen
option; score values are the expected level. Boolean answers include the
probability of true and use 0.5 to select their value.

Pass `images` (PIL images) or `videos` (frame arrays) to `predict` for
multimodal states, with optional `media_kwargs` for the processor.
`max_length` (default 16384) bounds the prompt and `max_state_tokens` the state.
