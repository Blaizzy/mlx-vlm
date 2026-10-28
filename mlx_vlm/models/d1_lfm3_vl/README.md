# d1 on MLX

LiquidAI's d1 decision models answer typed questions about a state (text, JSON,
pictures, audio, video) in one forward pass, read at answer slots: nothing is
decoded, so every answer has its declared shape. Two checkpoints load here:

| Checkpoint | Backbone | Package | Media |
|---|---|---|---|
| `LiquidAI/d1-2B` | LFM3-VL | `d1_lfm3_vl` | pictures, video |
| `LiquidAI/d1-omni-2B` | LFM3-Omni | `d1_lfm3_omni` | pictures, video, audio, spoken questions |

Both record `D1Model` in their config and route here from their LFM3
`model_type`. The LFM2-based d1-3B is still refused. The weights, towers,
sanitizers and quantization are the LFM3-VL and LFM3-Omni ones; this package
adds the set attention and the readout. The prompt, question, calibration,
pointer, box and parsing code is adapted from the code shipped with the
checkpoints under the LFM Open License v1.0 ([LICENSE](LICENSE)); it runs
without PyTorch.

d1 checkpoints do not generate text: `generate`, `stream_generate` and
`python -m mlx_vlm.generate` (text or audio) refuse them, and `mlx_vlm.server`
refuses to load them (HTTP 400; a preloaded `--model` stops at startup).

## Python

```python
from mlx_vlm import load

model, processor = load("path/to/d1-2B")  # attaches the processor and calibration.json

ticket = "I was charged twice and my parcel never arrived. Please refund one charge."
out = model.system_one(ticket, {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices",
                          "shipping": "Delivery and parcels",
                          "technical": "App or site faults"}},
    "topics": {"type": "choice", "multiple": True,
               "instructions": "Which topics does the message raise?",
               "criteria": {"billing": "Payments and charges",
                            "delivery": "Shipping and parcels",
                            "login": "Signing in to the account"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["low", "medium", "high"]},
})
a = out["answers"]
print(round(a["refund"]["noul"], 2), a["team"]["choice"], a["topics"]["choices"],
      a["urgency"]["legend"][str(int(a["urgency"]["score"]))])
# 1.0 billing ['billing', 'delivery'] medium
print(out["usage"])  # {'input_tokens': 189, 'output_tokens': 0}
```

`model.system_one(state, questions, media=None, score_readout="median",
set_size=None)` and `model.system_one_batch([(state, questions[, media]), ...])`
mirror the checkpoint's Transformers API. `state` is a string, any JSON value,
or `None` with media alone. `media` lists PIL images or
`{"type": "image" | "audio" | "video", "path": ...}` dicts, or the decoded value
under its type: `{"type": "audio", "audio": waveform, "sampling_rate": 16000}`,
`{"type": "video", "video": frames, "fps": 4.0}` (frames `(T, H, W, C)` or
`(T, C, H, W)`, sampled as the checkpoint's video processor samples them). A
spoken question (omni) has a media dict as its `instructions`. A box or boxes
question needs exactly one picture.

`model.engine.system_one(..., abstain_below=0.6)` picks a choice's
"I don't know" option when the best other option holds less than that share.
`D1_VISION_MAX_PIXELS` overrides the checkpoint's picture cap
(`d1_vision_max_pixels`), as in the reference.

### Answers

| Question | Answer |
|---|---|
| `noul` | `{"type": "noul", "noul": p_yes}` |
| `choice` | `{"type": "choice", "choice", "confidence", "probabilities"}` |
| `choice`, `multiple: true` | `{"type": "choice", "choices": [...], "probabilities"}` (each option's calibrated P(yes)) |
| `score` | `{"type": "score", "score", "confidence", "probabilities", "legend"}`; `score_readout` `median` (default), `mean` (unrounded) or `argmax` |
| `text` / `number` / `date` | `{"type", <type>: value, "span": [a, b], "found"}`, plus `text` (number, date), `unit` (number) and `path` (a JSON state's string); value `None` when not found |
| `number` with `range` (and `step`) | `{"type": "number", "number": median, "interval": [p10, p90], "unit"}` |
| `box` | `{"type": "box", "box": {"top", "bottom", "left", "right"} \| None, "found", "confidence"}` (fractions of the picture) |
| `boxes` (`max` 1-16) | `{"type": "boxes", "boxes": [{"box", "found", "confidence"}], "count", "at_least", "presence"}` |

Spans index the state as rendered: the string itself, or
`json.dumps(state, ensure_ascii=False, indent=2)`.

## Command line

```bash
python -m mlx_vlm.decide --model path/to/d1-2B \
  --state "I was charged twice. Please refund one charge." \
  --questions '{"refund": {"type": "noul", "instructions": "Is it a refund request?"}}'

python -m mlx_vlm.decide --model path/to/d1-omni-2B --audio call.wav \
  --questions-file questions.json --set-size 4 --score-readout mean
```

`--state` takes text, `--state-file` a file (a `.json` file is read as JSON),
`--questions` JSON, `--questions-file` a JSON file. `--image`, `--audio` and
`--video` repeat; the state's media are the pictures, then the clips, then the
videos. The answers print as JSON.

## How it runs

Questions are packed `set_size` at a time (the checkpoint's `d1_set_size`, 16)
after their state, one unpadded row per group: the prefix and the first
question are tokenized as one string, each further question as
`QUESTION:\n` plus its suffix. On full-attention layers a token sees the state
and its own question both ways (`(q < s) | (k < s) | (g[q] == g[k])`); sliding
layers stay causal in their 256-token window and the convolutions causal over
the row, as the checkpoint was trained. Positions run over the whole row.
The state reads every question in its group, so there is no prefix cache
between groups. Each state picture, clip or video is encoded once per request
and placed in every row. Only the option rows of the tied head are computed
at each slot; reads (text, number, date) point into the state's hidden states.

## Parity

MLX against the checkpoints' own Transformers code (`D1Model`, fp32, CPU) for
`LiquidAI/d1-2B` and `LiquidAI/d1-omni-2B`, on a Mac Studio (M3 Ultra), Metal. Requests: the model cards' ticket (all four untyped types,
`set_size` 16 and 1), invoice (text, number, date, a missing field), batch, the
COCO two-cats picture (estimate, box, 4 boxes), omni beeps; plus JSON states
with paths and special-token text, estimates on number and code grids, score
readouts, abstain, `set_size` 3, 16-block boxes, picture with a text state,
decoded video frames, and on omni a spoken question, a 44.1 kHz clip and a
picture with a clip. Every answer, span, box edge and `input_tokens` count
matched; the table gives the largest probability difference.

| | d1-2B | d1-omni-2B |
|---|---:|---:|
| fp32, card rows | 1.2e-06 | 3.1e-06 |
| fp32, extra rows | 4.3e-05 | 1.0e-05 |

Lower precisions against MLX fp32 on a fixed set of 52 (d1-2B) and 59 (omni)
questions: the card requests, JSON states with reads, estimates, score
readouts, `set_size` 1 and 3, pictures, boxes, video, and on omni audio and a
spoken question. Largest probability difference, and the answers that change:

| | d1-2B | d1-omni-2B |
|---|---|---|
| bf16 (as shipped) | 0.020, same answers; box edges ±0.001 | 0.019, same answers; box edges ±0.001 |
| 8-bit (`convert -q --q-bits 8`) | 0.019, same answers; box edges ±0.001 | 0.026, same answers; box edges ±0.001 |
| 4-bit (`--q-bits 4`) | 0.26; 3 flip: two estimates (median and interval), a score at `set_size=3`; box edges ±0.003 | 0.19; 7 flip: a text span lost, `team` at `set_size=1`, five estimate intervals; box edges ±0.005 |

bf16 still flips near ties on wider probes (an estimate's grid step, two
candidate spans 0.025 nats apart), as the checkpoint's own bf16 run does.

Not compared with the reference: video files (`--video PATH` decodes with
mlx-vlm's OpenCV loader at the processor's sampling), other `d1_attention`
modes (refused), and states long enough that the dense per-row mask matters
for memory.

## Quantization

`python -m mlx_vlm.convert --hf-path path/to/d1-2B --mlx-path d1-2B-8bit -q
--q-bits 8` works as for LFM3-VL and LFM3-Omni (omni keeps its audio modules
unquantized) and copies `calibration.json` and the `d1_*` config keys, so the
converted model answers the same way. 8-bit kept every answer above; 4-bit
moves probabilities by up to ~0.25 and flips close estimates, scores, spans and
choices: use 8-bit or bf16. `--quant-method awq` scales no layer on the LFM3
backbone (it looks for `mlp`, LFM3 has `feed_forward`; the log shows
`AWQ scaling applied: {'layers': 0, 'groups': 0}`), so it gives plain 4-bit.
