# Laya decisions on MLX

Use a prepared checkpoint with `model_type: "laya"` in its root `config.json`,
alongside `encoder_config`, `head_layers`, and `decision_config`. Copy the
published encoder config and `rl_agent_config.json` settings into those fields,
and place tokenizer files at the checkpoint root alongside the weights.
Each Laya variant should be prepared as its own checkpoint directory.

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/laya")
result = predict(model, processor, "Please refund my duplicate charge", {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": None, "technical": None, "sales": None},
    },
    "refund": {
        "type": "bool",
        "instructions": "Does the customer request a refund?",
    },
})
print(result["answers"]["department"]["value"])
```

Laya supports `choice`, `score`, and `bool` decisions; `noul` is an alias for
`bool`. A `score` question takes an ordered list of criteria. All questions in
one request are batched. The decision path does not generate tokens.

Each answer has `type` and `value`. Choice and score answers include calibrated
`probabilities`; Boolean answers include `probability`, the probability of true,
and use 0.5 to select their Boolean value. Laya's entropy confidence and action
probability are preserved in `metadata`; they are not interchangeable with
other models' confidence metrics. Model-specific tokenization, heads, and
calibration remain in the Laya implementation.

The shared `predict` function rejects unsupported decision types before
inference. Models supporting multi-label classification return independent
`scores`, which need not sum to one. Inspect `model.decision_types` for the
supported types. Loading uses the standard `lazy`, `strict`, and `revision`
options and supports converted checkpoints through the same path.
