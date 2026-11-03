# d1-3B decisions on MLX

Published checkpoint: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B).

```python
from mlx_vlm import load, predict

model, processor = load("LiquidAI/d1-3B")
result = predict(model, processor, "I was charged twice, please refund one.", {
    "refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds", "technical": "App faults"},
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
}, images=["http://images.cocodataset.org/val2017/000000039769.jpg"])
```

From the command line, `python -m mlx_vlm decide --model LiquidAI/d1-3B` takes
`--state` and/or one or more `--image` options; the server's `/v1/decisions`
takes `"images"` the same way.

The checkpoint is an LFM2.5-VL-3B finetune. Its `config.json` keeps
`model_type: "lfm2_vl"`, and the `auto_map.AutoModel` entry ending in `D1Model`
selects this package, so the original repository and its conversions load
unchanged. Chat generation, the processor, and conversion behave as for
LFM2-VL.

A decision is one forward pass that stops at the answer slot; no tokens are
generated. The prompt and readout follow the published System One defaults.
The state (a string, any JSON value, or `None` when images are the whole state)
and each question are rendered into the chat template. Each option scores the
best log-probability among its answer tokens (`yes`/`no` forms, level digits,
or single-token option codes), and the scores are softmaxed over the options
without calibration.

d1 supports `choice`, `score`, and `bool`; `noul` is a Boolean alias. Choice
criteria map labels to descriptions or list labels; score criteria are an
ordered list of 2 to 10 levels; Boolean criteria optionally describe `true` and
`false`, and other keys raise `ValueError`. Choice returns a label, score its
expected level, and Boolean the probability of `true`; `confidence` and the
score `legend` are in `metadata`.

`images` is one image or a list of PIL images, paths, or URLs; files are
turned upright by their EXIF orientation, PIL images are read as given. Each is
first downscaled to at most 1024 × 1024 pixels in area, then tiled by the
LFM2-VL processor. One question runs as a single pass. Several questions read the
state and its images once: the shared prefix fills a cache that each question
continues as its own row. A prompt or shared prefix over 2048 tokens is read in
chunks of 2048 tokens, or of `token_budget` if that is smaller, which bounds the
memory a long state takes. Questions run in batches of similar length, longest
first: a batch takes no question under half the length of its longest, so one
long question does not pad many short ones. `token_budget` (default 65536)
bounds the padded tokens per batch of questions, prefix included.
`usage.input_tokens` counts the prefix once plus each question's tokens.

The checkpoint runs in bfloat16, which moves probabilities by up to a few
hundredths, as in the PyTorch reference. In float32
(`model.set_dtype(mx.float32)`) they match the reference to about 1e-5 on text;
with images, the MLX image processor's resize differs slightly from
torchvision's, which moves probabilities by up to about 5e-3 on yes/no questions
and 1.6e-2 on questions with many options (POPE images, same top option).
4-bit conversions shift probabilities further, by up to about 0.32 in parity
checks.
