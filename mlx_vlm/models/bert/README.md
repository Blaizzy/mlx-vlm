# BERT

`bert` covers three heads over one encoder:

| Class | Architectures | Used for |
| --- | --- | --- |
| `Model` | `BertModel` | Sentence embeddings |
| `SequenceClassificationModel` | `BertForSequenceClassification` | Rerankers |
| `TokenClassificationModel` | `BertForTokenClassification` | NER, PII tagging |

## Token classification

Any `*ForTokenClassification` BERT checkpoint with an `id2label` map loads with
`load_token_classifier`, which tags text and returns character spans plus a
redacted copy.

```python
from mlx_vlm.token_classification import load_token_classifier

classifier = load_token_classifier("nativ-community/rampart-mlx-4bit")
result = classifier(
    "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net",
    keep_labels=("CITY", "STATE", "ZIP_CODE"),
)

print(result.spans)
print(result.redacted_text)
# I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>
```

The same pipeline backs `mlx_vlm.privacy_filter`, so `load_privacy_filter` and
`python -m mlx_vlm.privacy_filter` accept these checkpoints too.

Spans in `keep_labels` are still reported but left in `redacted_text`. Pass
`replacement="[REDACTED]"` for a single replacement string instead of typed
placeholders. Inputs longer than `max_position_embeddings` are classified in
windows.

```sh
python -m mlx_vlm.token_classification \
  --model nativ-community/rampart-mlx-4bit \
  --keep CITY,STATE,ZIP_CODE \
  "Call Maria Garcia on 617-555-0142."
```

### Rampart

[Rampart](https://ndstudio.gov/posts/say-hello-to-rampart) is National Design
Studio's MiniLM PII tagger (6 layers, 35 BIO labels, 7 Latin-script languages).
Upstream ships only a 4-bit ONNX export; the MLX checkpoints are repacked from
it without re-quantizing.

| Checkpoint | Size | Weights |
| --- | --- | --- |
| [`nationaldesignstudio/rampart`](https://huggingface.co/nationaldesignstudio/rampart) | 14.7 MB | Official ONNX q4 |
| [`nativ-community/rampart-mlx-4bit`](https://huggingface.co/nativ-community/rampart-mlx-4bit) | 15 MB | 4-bit linears, 8-bit embeddings |
| [`nativ-community/rampart-mlx-fp16`](https://huggingface.co/nativ-community/rampart-mlx-fp16) | 37 MB | Dequantized fp16 |

The upstream system also runs a regex layer for SSNs, payment cards and IP
addresses before the model. That layer is not part of this port, so apply your
own pattern matching for those classes.

Token classifiers are a redaction aid, not an anonymization or compliance
guarantee. Evaluate on the target domain before relying on them.
