# Clef decisions on MLX

Prepared checkpoints: Clef
([8-bit](https://huggingface.co/nativ-community/clef-MLX-8bit),
[NVFP4](https://huggingface.co/nativ-community/clef-MLX-NVFP4),
[MXFP4](https://huggingface.co/nativ-community/clef-MLX-MXFP4)) and Clef-Flash
([8-bit](https://huggingface.co/nativ-community/clef-flash-MLX-8bit),
[NVFP4](https://huggingface.co/nativ-community/clef-flash-MLX-NVFP4),
[MXFP4](https://huggingface.co/nativ-community/clef-flash-MLX-MXFP4)).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/clef-flash-MLX-8bit")
result = predict(model, processor, "Please refund my duplicate charge", {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": ["billing", "technical", "sales"],
    },
})
print(result["answers"]["department"]["value"])
```

Use a checkpoint with `model_type: "clef"` in its root `config.json` and a
`head_config` object containing the published `joint_head_config.json`, with
the `joint_head.safetensors` tensors listed in `model.safetensors.index.json`.
Weights use the existing Qwen3.5 sanitization.

Clef supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. All
questions are scored jointly in one forward pass, and no answer tokens are
generated. Choice and score answers include `probabilities`, Boolean answers
the probability of true. `images` and `videos` may be passed to `predict`.
