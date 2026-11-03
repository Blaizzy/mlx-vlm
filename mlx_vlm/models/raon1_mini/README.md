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

The implementation follows the [published Transformers reference](https://huggingface.co/KRAFTON/Raon-1-mini-preview/blob/8c9b802acacdbda9df16606689fcf437cf7262fb/modeling_raon1_mini.py). Shared checks cover language output, input embeddings, and full/chunked prefill and decode:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py -k raon
```
