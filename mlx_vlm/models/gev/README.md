# GEV decisions on MLX

Prepared checkpoints:
[8-bit](https://huggingface.co/nativ-community/GEV-26B-Decide-MLX-8bit),
[NVFP4](https://huggingface.co/nativ-community/GEV-26B-Decide-MLX-NVFP4),
[MXFP4](https://huggingface.co/nativ-community/GEV-26B-Decide-MLX-MXFP4).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/GEV-26B-Decide-MLX-8bit")
result = predict(model, processor, "The parcel arrived damaged and I want my money back.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
})
print(result["answers"]["refund"]["value"])
```

Use a checkpoint with `model_type: "gev"` in its root `config.json` and a
`decision_config` object containing the slot `ranges`, `softcap`, per-kind
`temperature_by_type`, and the System 1 LoRA's `lora_alpha` and `lora_r`. The
LoRA is merged into the backbone at load time.

GEV supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. Each
question is read in one forward pass, and no answer tokens are generated.
Choice questions take up to 256 options, scored as groups of at most 16 and a
final beyond 16. Score questions use the six levels 0 to 5, and Boolean
questions take no criteria. The state may be text, JSON, or a list mixing text
with images.
