# EmbeddingGemma 2 (work in progress)

Native MLX implementation for `gg-hf-em/embeddinggemma-2`, with a bidirectional
text encoder, per-layer projections, a 768-dimensional token projection, and
the shared Gemma 4 vision and audio towers. The model returns token embeddings
in `last_hidden_state` and normalized, mask-aware mean pooled embeddings in
`text_embeds`.

**Reference validation is pending.** The implementation was reconstructed from
the checkpoint configuration, weight shapes, documentation, and existing Gemma 4
components. Transformers 5.17.0 and public Transformers main did not contain
`EmbeddingGemma2Model` or `EmbeddingGemma2Processor` when this port was prepared.
The source branch providing those classes is required before this implementation
can be considered verified. The 20 Python examples in the
[upstream documentation](https://huggingface.co/gg-hf-em/embeddinggemma-2/blob/main/embedding_gemma2_documentation.md)
have **not** been validated end to end.

## Text retrieval

```python
import mlx.core as mx
from transformers import AutoTokenizer

from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.utils import get_model_path

path = get_model_path("gg-hf-em/embeddinggemma-2")
model = load_embedding_model(path)
tokenizer = AutoTokenizer.from_pretrained(path)
inputs = tokenizer(
    [
        "task: search result | query: Which planet is known as the Red Planet?",
        "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
        "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    ],
    padding=True,
    return_tensors="np",
)
output = model(**{key: mx.array(value) for key, value in inputs.items()})
embeddings = output.text_embeds
print(embeddings[:1] @ embeddings[1:].T)

# Matryoshka prefixes must be normalized again after truncation.
embeddings = embeddings[:, :256]
embeddings /= mx.linalg.norm(embeddings, axis=-1, keepdims=True)
```

The forward method also accepts `pixel_values`, `image_position_ids`,
`pixel_values_videos`, `video_position_ids`, `input_features`, and
`input_features_mask`. These must be paired with the corresponding expanded
media placeholders in `input_ids`; mismatched media token counts raise an error.
Use the upstream EmbeddingGemma2 processor once its implementation is available.

## Selective tower loading

```python
from mlx_vlm.utils import load_config

config = load_config(path)
config["audio_config"] = None
config["vision_config"] = None
text_model = load_embedding_model(path, config=config)
```

Omitted tower weights are discarded during sanitization. Converted MLX convolution
weights are accepted without applying the PyTorch-to-MLX transpose a second time.

## Validation completed so far

- Downloaded checkpoint revision `fc77679a26fcb86250765859d04ce2fcc6cb0b2c` and
  loaded all weights with strict checking on an Apple M5 Max.
- Text retrieval smoke test: the documented Mars passage ranks above Venus
  (cosine similarities approximately 0.853 and 0.677 with the BF16 checkpoint).
- A manually prepared text/image/audio/video batch produces four finite,
  normalized 768-dimensional embeddings. This checks the model tensor interface,
  not the unavailable EmbeddingGemma2 processor.
- Unit tests cover padding, bidirectional attention, media placement, disabled
  towers, sanitization idempotence, and strict checkpoint reloads.

Still required: comparison against the actual EmbeddingGemma2 reference,
confirmation of text attention/PLE semantics, all upstream Sentence Transformers
and AutoModel examples, processor ordering and video sampling controls, and
conversion through the complete model/processor loader.

Run the focused regression tests with:

```sh
python -m pytest -q mlx_vlm/tests/test_embedding_gemma2.py \
  mlx_vlm/tests/test_models.py -k embedding_gemma2
```
