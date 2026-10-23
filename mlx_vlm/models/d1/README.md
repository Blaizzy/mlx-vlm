# d1-3B decisions on MLX

Published checkpoint: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B).

```python
from mlx_vlm import load, predict

model, processor = load("LiquidAI/d1-3B")
result = predict(model, processor, "I was charged twice, please refund one.", [
    {
        "name": "refund",
        "type": "predicate",
        "instructions": "Is the customer asking for a refund?"
    },
    {
        "name": "team",
        "type": "choice",
        "instructions": "Which team should handle this?",
        "choices": [
            {
                "value": "billing",
                "description": "Charges, refunds"
            },
            {
                "value": "technical",
                "description": "App faults"
            }
        ]
    },
    {
        "name": "urgency",
        "type": "score",
        "instructions": "How urgent is this?",
        "levels": [
            {
                "label": "Can wait"
            },
            {
                "label": "Today"
            },
            {
                "label": "Blocking the customer now"
            }
        ]
    }
])
print(result["answers"][1]["choice"])

photo = predict(model, processor, None, [
    {
        "name": "cats",
        "type": "choice",
        "instructions": "How many cats are there?",
        "choices": [
            {
                "value": "one"
            },
            {
                "value": "two"
            },
            {
                "value": "more"
            }
        ]
    }
], images=["http://images.cocodataset.org/val2017/000000039769.jpg"])
```

From the command line, `python -m mlx_vlm decide --model LiquidAI/d1-3B` takes
`--input` and/or one or more `--image` options; the server's `/v1/decisions`
takes `"images"` the same way.

The checkpoint is an LFM2.5-VL-3B finetune. Its `config.json` keeps
`model_type: "lfm2_vl"`, and the `auto_map.AutoModel` entry ending in `D1Model`
selects this package, so the original repository and its conversions load
unchanged. Chat generation, the processor, and conversion behave as for
LFM2-VL.

A decision is one forward pass that stops at the answer slot; no tokens are
generated. The prompt and readout follow the published System One defaults.
The native state (text, JSON, or `None` when images carry the whole input)
and each question are rendered into the chat template. Each option scores the
best log-probability among its answer tokens (`yes`/`no` forms, level digits,
or single-token option codes), and the scores are softmaxed over the options
without calibration.

The shared Python API, CLI, and `/v1/decisions` use the
[OpenAI Decisions schema](../../../docs/decisions.md). Supply an ordered array
of `predicate`, `choice`, or `score` questions. Choice values have optional
descriptions; scores take 2–10 labeled levels. Answers are ordered, Boolean
answers carry `probability`, and choice/score answers include probability arrays
and the local confidence statistic. The native `model.predict()` and System One
adapter retain named questions, `criteria`, and `bool`/`noul`.

`images` is one image or a list of PIL images, paths, or URLs; files are
turned upright by their EXIF orientation, PIL images are read as given. Each is
first downscaled to at most 1024 × 1024 pixels in area, then tiled by the
LFM2-VL processor. One question runs as a single pass. Several questions read the
state and its images once: the shared prefix fills a cache that each question
continues as its own row. `token_budget` (default 65536) bounds the padded
tokens per batch of questions, prefix included. `usage.input_tokens` counts the
prefix once plus each question's tokens.

The checkpoint runs in bfloat16, which moves probabilities by up to a few
hundredths, as in the PyTorch reference. In float32
(`model.set_dtype(mx.float32)`) they match the reference to about 1e-5 on text;
with images, the MLX image processor's resize differs slightly from
torchvision's, which moves probabilities by up to about 1e-3.
4-bit conversions shift probabilities further, by up to about 0.2 in parity
checks.
