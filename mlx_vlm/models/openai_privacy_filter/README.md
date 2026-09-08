# OpenAI Privacy Filter

`openai/privacy-filter` is a bidirectional, sparse-MoE token classifier for
detecting and masking personally identifiable information. The MLX port loads
the original Hugging Face safetensors directly and includes the model's
constrained BIOES Viterbi decoder.

## Supported Models

| Source | Repository ID |
| --- | --- |
| Official | [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) |
| MLX BF16 | [`mlx-community/openai-privacy-filter-bf16`](https://huggingface.co/mlx-community/openai-privacy-filter-bf16) |
| MLX 4-bit | [`mlx-community/openai-privacy-filter-4bit`](https://huggingface.co/mlx-community/openai-privacy-filter-4bit) |
| MLX 5-bit | [`mlx-community/openai-privacy-filter-5bit`](https://huggingface.co/mlx-community/openai-privacy-filter-5bit) |
| MLX 6-bit | [`mlx-community/openai-privacy-filter-6bit`](https://huggingface.co/mlx-community/openai-privacy-filter-6bit) |
| MLX 8-bit | [`mlx-community/openai-privacy-filter-8bit`](https://huggingface.co/mlx-community/openai-privacy-filter-8bit) |
| MLX MXFP4 | [`mlx-community/openai-privacy-filter-mxfp4`](https://huggingface.co/mlx-community/openai-privacy-filter-mxfp4) |
| MLX NVFP4 | [`mlx-community/openai-privacy-filter-nvfp4`](https://huggingface.co/mlx-community/openai-privacy-filter-nvfp4) |
| MLX MXFP8 | [`mlx-community/openai-privacy-filter-mxfp8`](https://huggingface.co/mlx-community/openai-privacy-filter-mxfp8) |

## Python

```python
from mlx_vlm.privacy_filter import load_privacy_filter

detector = load_privacy_filter("openai/privacy-filter")
result = detector(
    "Alice Smith can be reached at alice@example.com.",
)

print(result.spans)
print(result.redacted_text)
```

The result contains character offsets, typed labels, source text for each
span, and redacted text. Pass `replacement="[REDACTED]"` to use a single
replacement string instead of typed placeholders such as `<PRIVATE_EMAIL>`.

Use independent per-token decoding only when explicitly needed:

```python
result = detector(text, decode="argmax")
```

The default Viterbi decoder enforces coherent BIOES spans and reads
`viterbi_calibration.json` from the checkpoint. Transition biases can be
overridden when loading the model.

## Command line

```sh
python -m mlx_vlm.privacy_filter \
  --model openai/privacy-filter \
  "Alice Smith can be reached at alice@example.com."
```

## Quantization

The MLX conversions listed above load directly. Their split expert projections
and the original checkpoint's fused expert projections are both supported:

```python
detector = load_privacy_filter("mlx-community/openai-privacy-filter-8bit")
```

The standard converter can produce an MLX-native quantized checkpoint:

```sh
mlx_vlm.convert \
  --hf-path openai/privacy-filter \
  --mlx-path privacy-filter-4bit \
  --quantize \
  --q-bits 4
```

The attention sink tensors remain FP32, and router projections default to
8-bit during mixed model quantization.

Privacy Filter is a redaction aid, not an anonymization or compliance
guarantee. Evaluate it on the target domain and retain review paths for
high-sensitivity workflows.
