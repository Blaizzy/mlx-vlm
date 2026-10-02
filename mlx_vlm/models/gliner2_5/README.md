# GLiNER2.5

Native MLX inference for the Fastino GLiNER2.5 boundary-extractor checkpoints:

- `fastino/gliner2.5-small-v1`
- `fastino/gliner2.5-base-v1`
- `fastino/gliner2.5-multi-v1`

Prepared small checkpoint: [nativ-community/gliner2.5-small-v1](https://huggingface.co/nativ-community/gliner2.5-small-v1).

```python
from mlx_vlm import load

model, processor = load("nativ-community/gliner2.5-small-v1")

entities = model.extract_entities(
    processor,
    "Apple hired Sam in New York.",
    ["company", "person", "location"],
    include_confidence=True,
    include_spans=True,
)

sentiment = model.classify_text(
    processor,
    "The launch was excellent.",
    {"sentiment": ["positive", "neutral", "negative"]},
)
```

Entity label descriptions can be supplied as a mapping instead of a list. For
multilingual text without whitespace-delimited words, pass
`word_splitter="char"` to the inference method.

This initial native port covers the shared-boundary entity extraction and text
classification paths used by the collection. Relation and record decoding are
rejected instead of silently returning incomplete structured output.

GLiNER2.5 is a bidirectional encoder extractor. Autoregressive speculative
decoding methods, including MTP and DFlash, do not apply to this architecture.

## Decision classification

The prepared checkpoint is [nativ-community/GLiNER2.5-Decide](https://huggingface.co/nativ-community/GLiNER2.5-Decide).

For an original checkpoint, prepare it with `model_type: "gliner2_5"`, `architecture: "span"`,
and the contents of `encoder_config/config.json` embedded as `encoder_config`
in root `config.json`. Keep the weights and tokenizer files at the root;
loading uses the standard loader without additional sidecar detection.

`fastino/GLiNER2.5-Decide` is a span-architecture checkpoint. Its classification
path is supported; its span and counting heads are unused by classification and
are omitted when loading. Entity extraction from span checkpoints is not yet
implemented and raises a clear error. Existing boundary models retain entity
extraction support.

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/GLiNER2.5-Decide")
result = predict(model, processor, "Please refund my duplicate charge", {
    "route": {"type": "choice", "criteria": ["billing", "technical"]},
    "tags": {
        "type": "multi_label",
        "criteria": ["refund", "login", "payment"],
        "threshold": 0.4,
    },
})
```

The shared decision contract uses `type` and `value` in each answer. Single-label
choices return softmax `probabilities`; multi-label decisions return independent
sigmoid `scores`. GLiNER supports `choice` and `multi_label`; use named choice
labels for Boolean or ordinal classification rather than requesting Laya's or
Decider's calibrated Boolean/expected-score readout. Optional `instructions`
become the GLiNER task prompt, and a criteria dictionary supplies label descriptions.
All tasks in one request share an encoder forward pass. For span checkpoints,
when no multi-label score meets the threshold, the highest-scoring label is
returned, matching the reference implementation.

The existing `classify_text(processor, text, tasks)` API remains available.
`return_scores=True` returns every label's probability or independent score.
