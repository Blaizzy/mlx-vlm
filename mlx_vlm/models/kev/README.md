# Kev decisions on MLX

[Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) is a full fine-tune of
Qwen3.8-27B with a pointer head. Convert the original release directly:

```shell
python -m mlx_vlm.convert --hf-path jaredpalmer/kev-27b \
    --mlx-path kev-27b-MLX-8bit -q --q-bits 8
```

Reading the release's `head.pt` needs PyTorch; converted checkpoints do not.
The 8-bit conversion peaks at 31 GB with an 8k-token state.

```python
from mlx_vlm import load, predict

model, processor = load("kev-27b-MLX-8bit")
result = predict(model, processor, "I was charged twice for order 1182.", {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries"},
    },
    "urgent": {"type": "bool", "instructions": "Does this need a reply today?"},
})
print(result["answers"]["team"]["value"])
```

Requests are encoded as the release encodes them: the state runs once, and
each question continues it as its own causal row, so questions never see one
another. The head scores each option against the question and applies the
release's calibration temperature; no answer tokens are generated.

Kev supports `choice` (options with optional descriptions), `score` (ordered
levels) and `bool`; `noul` is a Boolean alias, with optional descriptions keyed
by `false` and `true`. Choice and score answers include `probabilities`;
`metadata.confidence` is the release's confidence statistic, and a score's
`value` is its expected level index. As in the release, a state is limited to
65,536 tokens and a state plus one question to 8,192 more. Kev reads text only.
