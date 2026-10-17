# Decider-2b decisions on MLX

Prepared checkpoint: [nativ-community/decider-2b](https://huggingface.co/nativ-community/decider-2b).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/decider-2b")
result = predict(model, processor, "Please refund my duplicate charge", [
    {
        "name": "department",
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "choices": [
            {
                "value": "billing"
            },
            {
                "value": "technical"
            },
            {
                "value": "sales"
            }
        ]
    }
])
print(result["answers"][0]["choice"])
```

Use a checkpoint with `model_type: "decider2"` in its root `config.json`,
alongside the Qwen3.5 configuration and a `decision_config` object containing
the published `decider_config.json` settings. Tokenizer files belong at the
checkpoint root. This metadata must be prepared before loading the original
Mapika checkpoint or an older conversion labeled `qwen3_5_text`.

Weights use the existing Qwen3.5 text backbone and sanitization. No special
loader or runtime sidecar detection is required.

The shared `predict(model, processor, input, questions)` API, CLI, and
`/v1/decisions` use the [OpenAI Decisions schema](../../../docs/decisions.md).
Questions are an ordered array of `choice`, `score`, or `predicate` objects.
Answers contain the selected typed `choice`, the expected level index in `score`,
or the probability of true in `probability`. Choice and score answers include
probability arrays and the local confidence statistic. No answer tokens are generated.

Choices contain 2–255 values with optional descriptions. Scores contain 2–10
ordered levels with labels and optional descriptions. The native model interface
continues to use named questions with `criteria` and `bool`/`noul` types.

`independent=True` evaluates questions separately. Score levels are isolated
according to the checkpoint settings unless overridden with `isolated`.
`independent=False` packs questions together, allowing later questions to attend
to earlier ones. Per-type temperatures use checkpoint settings with the global
temperature as fallback; isolated score rows use the score temperature.

`max_state_tokens` defaults to 32768. Inference performs a full forward pass,
so long inputs may exceed device memory.
