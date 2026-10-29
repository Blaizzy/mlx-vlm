# Raon-1 Mini

Native text generation for [KRAFTON/Raon-1-mini-preview](https://huggingface.co/KRAFTON/Raon-1-mini-preview), using the `raon1_mini` model type.

The model combines 18 Raon Delta Attention layers, seven NoPE latent-attention layers, four residual streams, and SiTU expert feed-forward layers. Recurrent state stays in float32. The original checkpoint's MXFP8 experts remain quantized; its dense layers and output head retain their published precision.

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("KRAFTON/Raon-1-mini-preview", trust_remote_code=False)
prompt = apply_chat_template(
    processor,
    model.config,
    "What is 17 + 25? Reply with only the number.",
    enable_thinking=False,
)
result = generate(model, processor, prompt, max_tokens=128, temperature=0.0)
print(result.text)
```

Loading uses the built-in implementation without remote model code. The tokenizer's chat template controls thinking mode.

The sanitizer combines the source checkpoint's separate convolution and expert tensors, splits latent-attention projections, and packs E4M3/E8M0 expert tensors into native MXFP8 storage. Re-sanitizing saved native weights preserves their values.

The implementation follows the [published Transformers reference](https://huggingface.co/KRAFTON/Raon-1-mini-preview/blob/8c9b802acacdbda9df16606689fcf437cf7262fb/modeling_raon1_mini.py). Shared model tests cover numerical operators, full/chunked prefill and decode, padded batches, recurrent cache rollback, checkpoint loading, and quantized save/load round trips:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py mlx_vlm/tests/test_cache.py -k raon
```

A separate two-layer comparison against the pinned Transformers implementation covered dense and expert layers, randomized residual selectors, and chunk sizes 1, 3, and 8. All top tokens matched. Maximum logit error was `2.4e-7` in float32 and `0.00390625` in BF16.

DFlash2, the optional two-stage output head, and full-length 262K context validation are outside this implementation's validation scope.
