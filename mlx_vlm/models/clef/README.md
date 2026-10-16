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

Standard loading accepts prepared `model_type: "clef"` checkpoints with
`head_config`, original Cloudflare releases with `joint_head_config.json` and a
separate `joint_head.safetensors`, and earlier MLX conversions using
`joint_head_config`. The head is required even when the backbone shard index
does not list it. Sanitization accepts the published fused attention weights,
earlier MLX head layouts, and the current split projection layout.

Clef supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. All
questions are scored jointly in one forward pass, and no answer tokens are
generated. Choice and score answers include `probabilities`, Boolean answers
the probability of true. `images` and `videos` may be passed to `predict`.

See [shared decision serving](../../../docs/decisions.md) for scheduling,
state-prefix caching, cancellation, JSONL evaluation, and the TypeSafe adapter.
The model retains its native `predict()` and `/v1/decisions` answer schema.

## Quantization

```sh
python -m mlx_vlm convert --hf-path Cloudflare/clef-flash \
  --mlx-path ./clef-flash-4bit -q --q-bits 4 --q-group-size 64 \
  --decision-quantization preserve-output
```

`preserve-output` (the default for Clef) keeps the head and the lexical output
embeddings floating point. `backbone` also quantizes the output embeddings to
save memory while retaining the floating-point head. These exclusions apply
when selecting mixed recipes or custom quantization predicates. Vocabulary row
lookup gathers the required rows before dequantization; it does not dequantize
the full vocabulary matrix. Existing quantized checkpoints keep their stored
quantization layout when loaded.

Quantization can change probabilities and selected answers. Use the CLI
comparison report with representative requests before choosing a checkpoint.
