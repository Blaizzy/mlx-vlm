# d1-omni decisions on MLX

[LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) is a
decision model: a bidirectional LFM2.5 encoder trunk with a two-layer decision
head, a SigLIP2 vision tower, and a FastConformer audio encoder whose projected
features enter the trunk as a prefix in front of the text. It answers typed
questions about a state without generating tokens. The published checkpoint
loads directly; no remote code is needed.

```python
from mlx_vlm import load, predict

model, processor = load("LiquidAI/d1-omni-600M")
result = predict(model, processor, "I was charged twice, please refund one.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
})
print(result["answers"]["team"]["value"])

photo = predict(model, processor, None, {
    "cats": {"type": "choice", "instructions": "How many cats are there?",
             "criteria": ["one", "two", "more"]},
}, images=["cats.jpg"])

speech = predict(model, processor, None, {
    "speaking": {"type": "bool", "instructions": "Is anyone speaking in this clip?"},
}, audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

From the command line, `python -m mlx_vlm decide --model LiquidAI/d1-omni-600M`
takes `--state` and/or either one or more `--image` options or one `--audio`;
the server's `/v1/decisions` takes `"images"` or `"audio"` (a URL, path, or
base64 data URI) the same way.

d1-omni supports `choice`, `score`, and `bool`; `noul` is an alias for `bool`.
A `choice` takes `{label: description}` or a list of labels, a `score` an
ordered list of 2 to 10 levels, and a `bool` optional `{"false": ..., "true":
...}` descriptions. Choice and score answers include `probabilities` and
`metadata.confidence`; scores also carry the level `legend`. Boolean answers
include `probability`, the probability of true, and use 0.5 to select their
value.

A state is a string, any JSON value, or `None` when images or audio are the
whole state. `images` is one image or a list of PIL images, paths, or URLs,
read in order; files are turned upright by their EXIF orientation, PIL images
are read as given, and large images are tiled into up to ten 512 px tiles plus
a thumbnail. `audio` is one clip: 16 kHz mono samples (a NumPy array, list, or
`mx.array`; int16 PCM is scaled by 1/32768, floats are read as is), or a path,
URL, or binary file object of an audio file, which is downmixed and resampled
to 16 kHz; a file that fails to decode or holds no samples raises `ValueError`.
Clips are cut to 30 s and padded with silence to 0.5 s, and every 80 ms is one
prefix position. A request carries images or audio, not both.

Text answers are calibrated with the per-type temperatures in `config.json`.
Image and audio answers are the model's softmax as trained. With images, Boolean
questions without criteria get plain yes/no options; given criteria are written
as text, and null-valued ones fall back to the text wording, as in the reference.
Each question reads one text row (start token, state, and question) after the
media prefix, capped at 896 tokens after images, 15360 after audio,
and 16384 minus the prefix in every case. The instructions keep up to B =
max(96, min(24 × options + 32, cap / 2)) tokens, the options share another B,
and the state is cut on the right to the room that is left. After audio, as in
training, a missing state reads as `{}`, choices are written as numbered
options, and Boolean criteria are ignored. `usage.input_tokens` counts every
position the trunk reads, so each question counts the media prefix again. The
questions of a request run in length-sorted batches; `token_budget` (default
65536) bounds the padded positions per batch, media prefix included.

The checkpoint is float32 and runs in float32 by default, matching the PyTorch
reference to about 1e-5 in probability on text, images, and speech. For images
this holds against torchvision's native uint8 resize, which the reference uses on
Apple Silicon from torchvision 0.27 and on x86 with AVX2; on older or GPU setups
it resizes in float and its own answers move by up to about 0.08.
`--dtype float16` conversions keep the answers apart from near ties.
Four-bit quantization changes a noticeable share of answers, so prefer 8-bit
(`-q --q-bits 8`) or float16; the vision tower and the decision scorer stay
unquantized. The audio encoder is quantized with the trunk: keeping it in full
precision measured no gain at 4 or 8 bits, where the trunk dominates the drift,
and adds about 300 MB. Digital silence (all-zero samples) is numerically
ill-conditioned in the log-mel front end; the reference itself answers it
differently on CPU and GPU.
