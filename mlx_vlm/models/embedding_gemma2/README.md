# EmbeddingGemma 2

Multimodal embeddings for text, images, audio, video, and composed inputs on
Apple Silicon. Outputs are normalized 768-dimensional vectors, with Matryoshka
support for 128, 256, and 512 dimensions.

See the [examples notebook](../../../examples/embedding_gemma2.ipynb) for
retrieval, batching, all four modalities, mixed inputs, and conversion.

## Quick start

Requires MLX >= 0.32.3 and Transformers >= 5.14.0. The local processor does not
require Torch, TorchVision, TorchAudio, TorchCodec, or PyAV.

```python
import mlx.core as mx
from mlx_vlm import load

model, processor = load("nativ-community/embeddinggemma-2")
inputs = processor.tokenizer(
    [
        "task: search result | query: Which planet is known as the Red Planet?",
        "title: none | text: Venus is often called Earth's twin.",
        "title: none | text: Mars is known as the Red Planet.",
    ],
    padding=True,
    return_tensors="np",
)
embeddings = model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
print(embeddings[:1] @ embeddings[1:].T)  # Cosine similarities

# Optional: truncate and renormalize. Use the same dimension for all inputs.
embeddings = embeddings[:, :256]
embeddings /= mx.linalg.norm(embeddings, axis=-1, keepdims=True)
```

For media, use `processor(...)` or `processor.apply_chat_template(...)` with
`return_tensors="mlx"`, then call `model(**inputs).text_embeds`.

## Notes

- Video uses Metal color conversion with compatible OpenCV-bundled FFmpeg 7
  libraries. Unsupported builds and formats fall back to OpenCV; numerical
  agreement with the reference can differ.
- Conversion preserves the tokenizer, chat template, and processor settings.
  Reload converted directories with the same `load` API.
- BF16 and 8-bit passed numerical accuracy checks; standard 4-bit did not.
