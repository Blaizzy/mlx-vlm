# JEV decisions on MLX

Prepared checkpoints:
[8-bit](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-8bit),
[NVFP4](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-NVFP4),
[MXFP4](https://huggingface.co/nativ-community/JEV-27B-VL-MLX-MXFP4).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/JEV-27B-VL-MLX-8bit")
result = predict(model, processor, "The parcel arrived damaged and I want my money back.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
})
print(result["answers"]["refund"]["value"])
```

Use a checkpoint with `model_type: "jev"` in its root `config.json` and a
`decision_config` object containing the slot `ranges`, `verbalizer_ids`, and
`bias` from `decision_head.json`, per-kind `temperature_by_type`, and the
System 1 LoRA's `lora_alpha` and `lora_r`. The LoRA, including its `lm_head`
update, is merged into the backbone at load time.

JEV supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. Each
question is read in one forward pass from the option tokens' logits, and no
answer tokens are generated. Choice questions take up to 256 options in a single
pass. Score questions use the six levels 0 to 5, and Boolean questions take no
criteria. The state may be text, JSON, or a list mixing text with images.
