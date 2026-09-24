# BERT

`bert` implements the BERT encoder (`model_type: bert`), which also covers
MiniLM and other BERT-architecture checkpoints. One encoder is shared by three
task heads, selected from the checkpoint's `architectures`:

| Class | Architectures | Task | Loader |
| --- | --- | --- | --- |
| `Model` | `BertModel` | Sentence embeddings | `mlx_vlm.embedding_loader.load_embedding_model` |
| `SequenceClassificationModel` | `BertForSequenceClassification` | Reranking | `mlx_vlm.reranker_loader.load_reranker` |
| `TokenClassificationModel` | `BertForTokenClassification` | Per-token tagging (NER, PII detection) | `mlx_vlm.token_classification.load_token_classifier` |

## Token classification

Any BERT `*ForTokenClassification` checkpoint with BIO labels in `id2label`
loads with `load_token_classifier`. The classifier tags text and returns
character spans with a redacted copy:

```python
from mlx_vlm.token_classification import load_token_classifier

classifier = load_token_classifier("<bert-token-classification-checkpoint>")
result = classifier("text to tag")

print(result.spans)          # label, start, end, text for each entity
print(result.redacted_text)  # entities replaced by <LABEL> placeholders
```

- Sub-word pieces are grouped into words before spans are built.
- Inputs longer than `max_position_embeddings` are classified in windows.
- `keep_labels=("LABEL", ...)` reports those spans but leaves them in `redacted_text`.
- `replacement="[REDACTED]"` uses one replacement string instead of typed placeholders.

```sh
python -m mlx_vlm.token_classification --model <checkpoint> "text to tag"
```

The same pipeline backs `mlx_vlm.privacy_filter`, so `load_privacy_filter` and
`python -m mlx_vlm.privacy_filter` accept these checkpoints too.

## Example: PII redaction with Rampart

[Rampart](https://ndstudio.gov/posts/say-hello-to-rampart) is one use of the
token-classification head. It is National Design Studio's PII tagger, a
6-layer MiniLM with 35 BIO labels in 7 Latin-script languages. Upstream ships
only a 4-bit ONNX export; the MLX checkpoints are repacked from it without
re-quantizing.

| Checkpoint | Size | Weights |
| --- | --- | --- |
| [`nationaldesignstudio/rampart`](https://huggingface.co/nationaldesignstudio/rampart) | 14.7 MB | Official ONNX q4 |
| [`nativ-community/rampart-mlx-4bit`](https://huggingface.co/nativ-community/rampart-mlx-4bit) | 15 MB | 4-bit linears, 8-bit embeddings |
| [`nativ-community/rampart-mlx-fp16`](https://huggingface.co/nativ-community/rampart-mlx-fp16) | 37 MB | Dequantized fp16 |

Rampart's policy detects city, state and ZIP code but keeps them in the output:

```python
from mlx_vlm.token_classification import load_token_classifier

classifier = load_token_classifier("nativ-community/rampart-mlx-4bit")
result = classifier(
    "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net",
    keep_labels=("CITY", "STATE", "ZIP_CODE"),
)

print(result.redacted_text)
# I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>
```

```sh
python -m mlx_vlm.token_classification \
  --model nativ-community/rampart-mlx-4bit \
  --keep CITY,STATE,ZIP_CODE \
  "Call Maria Garcia on 617-555-0142."
```

The upstream Rampart system also runs a regex layer for SSNs, payment cards and
IP addresses before the model. That layer is not part of this port, so apply
your own pattern matching for those classes.

Token classifiers are a redaction aid, not an anonymization or compliance
guarantee. Evaluate on the target domain before relying on them.
