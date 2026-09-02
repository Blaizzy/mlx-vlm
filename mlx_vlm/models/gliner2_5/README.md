# GLiNER2.5

Native MLX inference for the Fastino GLiNER2.5 boundary-extractor checkpoints:

- `fastino/gliner2.5-small-v1`
- `fastino/gliner2.5-base-v1`
- `fastino/gliner2.5-multi-v1`

```python
from mlx_vlm.gliner import load_gliner

extractor = load_gliner("fastino/gliner2.5-small-v1")

entities = extractor.extract_entities(
    "Apple hired Sam in New York.",
    ["company", "person", "location"],
    include_confidence=True,
    include_spans=True,
)

sentiment = extractor.classify_text(
    "The launch was excellent.",
    {"sentiment": ["positive", "neutral", "negative"]},
)
```

Entity label descriptions can be supplied as a mapping instead of a list. For
multilingual text without whitespace-delimited words, load the model with
`word_splitter="char"`.

This initial native port covers the shared-boundary entity extraction and text
classification paths used by the collection. Relation and record decoding are
rejected instead of silently returning incomplete structured output.

GLiNER2.5 is a bidirectional encoder extractor. Autoregressive speculative
decoding methods, including MTP and DFlash, do not apply to this architecture.
