# Decider-2b decisions on MLX

```python
from mlx_vlm import load, predict

model, processor = load("path/to/mlx-decider-2b")
result = predict(model, processor, "Please refund my duplicate charge", {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": ["billing", "technical", "sales"],
    },
})
print(result["answers"]["department"]["value"])
```

Use a checkpoint with `model_type: "decider2"` in its root `config.json`,
alongside the Qwen3.5 configuration and a `decision_config` object containing
the published `decider_config.json` settings. Tokenizer files belong at the
checkpoint root. This metadata must be prepared before loading the original
Mapika checkpoint or an older conversion labeled `qwen3_5_text`.

Weights use the existing Qwen3.5 text backbone and sanitization. No special
loader or runtime sidecar detection is required.

The shared `predict(model, processor, state, questions)` API returns named
answers with `type` and `value`. Decider supports `choice`, `score`, and `bool`;
`noul` is a Boolean alias. Choice returns a label and score returns an expected
level index, both with `probabilities`. Boolean answers include the probability
of true. Confidence, certainty, and isolated-level metrics remain in `metadata`.
Unsupported question types fail before inference. No answer tokens are generated.

Choice criteria are 2–255 labels or a mapping of labels to descriptions. Score
criteria are an ordered list of 2–10 level descriptions. Boolean criteria are
optional descriptions keyed by `false` and `true`.

`independent=True` evaluates questions separately. Score levels are isolated
according to the checkpoint settings unless overridden with `isolated`.
`independent=False` packs questions together, allowing later questions to attend
to earlier ones. Per-type temperatures use checkpoint settings with the global
temperature as fallback; isolated score rows use the score temperature.

`max_state_tokens` defaults to 32768. Inference performs a full forward pass,
so long inputs may exceed device memory.
