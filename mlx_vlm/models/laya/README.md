# Laya decisions on MLX

Use a prepared checkpoint with `model_type: "laya"` in its root `config.json`,
alongside `encoder_config`, `head_layers`, and `decision_config`. Copy the
published encoder config and `rl_agent_config.json` settings into those fields,
and place tokenizer files at the checkpoint root alongside the weights.
Each Laya variant should be prepared as its own checkpoint directory.

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/laya")
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
    },
    {
        "name": "refund",
        "type": "predicate",
        "instructions": "Does the customer request a refund?"
    }
])
print(result["answers"][0]["choice"])
```

The public API supports `choice`, `score`, and `predicate` questions. All
questions in one request are batched. The decision path does not generate tokens.

The shared `predict()` API, CLI, and `/v1/decisions` use the
[OpenAI Decisions schema](../../../docs/decisions.md): ordered answers, typed
choice values, score levels, and predicate probabilities. Choice and score
answers include calibrated probability arrays and the local confidence statistic.
Tokenization, heads, and calibration remain in the Laya implementation.

The shared `predict` function rejects unsupported decision types before
inference. Use separate Boolean questions for independent labels. Inspect
`model.decision_types` for the supported types. Loading uses the standard `lazy`, `strict`, and `revision`
options and supports converted checkpoints through the same path.
