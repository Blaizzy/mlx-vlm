# Decider-2b decisions on MLX

Load `nativ-community/decider-2b` with the general loader. It is [`Mapika/decider-2b`](https://huggingface.co/Mapika/decider-2b) v11 with its weights renamed to the standard `model.*` layout, and gives the same outputs. It uses the existing Qwen3.5 text backbone and scores only the allowed option tokens at answer slots.

```python
from mlx_vlm import load
from mlx_vlm.models.decider2 import Decider2

model, processor = load("nativ-community/decider-2b")
decider = Decider2(model, processor)
result = decider.predict(
    {"ticket": "Please refund my duplicate charge"},
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": None, "technical": None, "sales": None},
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        },
    },
)
print(result["answers"])
```

The default `independent=True` runs each question in its own row and isolates score levels, matching the published inference API. `independent=False` packs questions into one forward pass; later questions can attend to earlier question text and may have different probabilities. Choice supports up to 255 options, score supports 2 to 10 levels, and `noul` reports the probability of yes. This is a non-generative decision API, separate from VLM chat and token generation.
