# Clef decisions on MLX

[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) pair a
Qwen3.5 vision-language backbone with a joint schema head that scores every
option of every question in a single forward pass.

```python
from mlx_vlm import load, predict

model, processor = load("path/to/clef")
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

Use a checkpoint with `model_type: "clef"` in its root `config.json` and a
`head_config` object holding the published `joint_head_config.json`. The
`joint_head.safetensors` tensors must be listed in `model.safetensors.index.json`
alongside the backbone shards. Weights use the existing Qwen3.5 sanitization;
the head is kept unquantized by `mlx_vlm.convert`.

Clef supports `choice`, `score`, and `bool`; `noul` is a Boolean alias.
Instructions are optional and default to the question name. Choice criteria are
labels or a mapping of labels to descriptions, score criteria are an ordered list
of level descriptions, and Boolean criteria optionally describe `true` and
`false`. Choice and score answers include `probabilities` and a `confidence` in
`metadata`; score values are the expected level. Boolean answers include the
probability of true.

Pass `images` (PIL images) or `videos` (frame arrays) to `predict` for
multimodal states, with optional `media_kwargs` for the processor.
`max_length` (default 16384) bounds the prompt and `max_state_tokens` the state.
No answer tokens are generated.
