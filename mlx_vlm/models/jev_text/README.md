# JEV-9B decisions on MLX

Prepared checkpoints:
[8-bit](https://huggingface.co/nativ-community/JEV-9B-MLX-8bit),
[NVFP4](https://huggingface.co/nativ-community/JEV-9B-MLX-NVFP4).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/JEV-9B-MLX-8bit")
result = predict(model, processor, "The parcel arrived damaged and I want my money back.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
})
print(result["answers"]["refund"]["value"])
```

`jev_text` is the text-only JEV model on the Qwen3.5 text backbone. It uses the
same decisions, options, and checkpoint preparation as [`jev`](../jev/README.md),
with `model_type: "jev_text"`. States may be text or JSON; images are not
accepted.
