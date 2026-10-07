# JEV decisions on MLX

JEV ([autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL),
[JEV-27B](https://huggingface.co/autotrust/JEV-27B),
[JEV-9B](https://huggingface.co/autotrust/JEV-9B)) answers typed questions with
a calibrated probability for every option in one forward pass. The base model is
the unchanged Qwen3.8-27B or Qwen3.5-9B (System 2); a LoRA adapter and a small
decision head form System 1.

Prepared checkpoint: [Bayway/JEV-27B-VL-MLX-4bit](https://huggingface.co/Bayway/JEV-27B-VL-MLX-4bit)
(about 17 GB of unified memory). JEV-27B and JEV-27B-VL ship the same adapter
and decision head, so this checkpoint answers for both.

```python
from mlx_vlm import load, predict

model, processor = load("Bayway/JEV-27B-VL-MLX-4bit")
result = predict(model, processor, "Customer: my card was charged twice for one coffee.", {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": ["billing", "shipping", "tech support"],
    },
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
})
print(result["answers"]["team"]["probabilities"])
```

The state is a string, a JSON object, or a list that mixes text with images:

```python
state = ["Listing photo: ", {"image": "item.jpg"}, "\nSeller title: wireless earbuds"]
```

Images can be paths, URLs, data URLs, or PIL images. They are encoded once per
request and shared by all questions.

JEV supports `choice`, `bool` (alias `noul`), and `score`:

- `choice` takes 2–256 labels, or labels mapped to one-line descriptions
  (`"billing": "use when the customer disputes a charge"`). The first 16 options
  use the trained decision slots; more options continue the labels (Q–Z, then
  AA, AB, …) and are reported with `adaptation: "wide-labels"`.
- `bool` takes no criteria descriptions. Phrase the instructions so that true is
  the outcome you want the probability of.
- `score` is JEV's fixed 0–5 scale. Give six level descriptions; JEV's prompt
  has no place for them, so they do not steer the model and are only returned
  as `metadata.legend`. State the scale in the instructions ("Rate urgency on a
  0-5 scale.").

Each question is rendered with JEV's bare template, the final-norm hidden state
of its last token is read by the decision head, and the head's bias and the
per-type temperature from `calibration.json` are applied before the softmax. No
tokens are generated. Each question is one forward pass, so its answer does not
depend on the other questions in the request. States longer than
`max_state_tokens` (default 32768, counting image tokens) are rejected; pass
`max_state_tokens=None` to `predict()` to lift the limit.

Decisions on one loaded model run one at a time, because the System 1 switch is
shared by the model. For the same reason, do not call `generate()` on a model
object while a decision on it is in progress.

System 2 is the base model itself: `generate` uses the same checkpoint with the
adapter switched off, so its output is unchanged by JEV.

Prepare a checkpoint from an MLX base and a JEV adapter:

```sh
python -m mlx_vlm.models.jev.convert \
  --model mlx-community/Qwen3.8-27B-4bit \
  --adapter autotrust/JEV-27B-VL \
  --mlx-path JEV-27B-VL-4bit
```

The base weights are copied unchanged. The adapter stays unmerged on top of
them, so a quantized base keeps the adapter in full precision, and the decision
rows are read from the base `lm_head` into a float32 head. Use the same command
for JEV-27B; for JEV-9B use an MLX Qwen3.5-9B base, with `--adapter-subfolder
vl/adapter_vllm` (the same weights, named for the multimodal model).
