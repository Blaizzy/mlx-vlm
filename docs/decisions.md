# Decision serving and batch evaluation

`mlx_vlm.predict(model, processor, input, questions)`, `mlx_vlm.decide`, and
`POST /v1/decisions` target the [OpenAI Decisions schema](https://developers.openai.com/api/reference/resources/decisions/methods/create).
They share model loading, inference, and response formatting. Clef, Decider,
Laya, and the LiquidAI d1 models use this interface. Each model declares its
supported question types and media. There is one public schema, without a
format selector; `/v1/systemone` is a separate compatibility adapter.

Save this as `request.json`:

```json
{
  "model": "Cloudflare/clef-flash",
  "input": "The checkout service is down",
  "questions": [
    {"name": "outage", "type": "predicate", "instructions": "Is there a service outage?"},
    {"name": "team", "type": "choice", "instructions": "Choose the responsible team", "choices": [
      {"value": "technical", "description": "Service faults"},
      {"value": "billing", "description": "Payments and refunds"}
    ]},
    {"name": "urgency", "type": "score", "instructions": "Rate urgency", "levels": [
      {"label": "Low"}, {"label": "Medium"}, {"label": "High", "description": "Blocking customers"}
    ]}
  ]
}
```

```python
import json
from mlx_vlm import load, predict

request = json.load(open("request.json"))
model, processor = load(request["model"])
result = predict(model, processor, request["input"], request["questions"])
print(result["answers"][1]["choice"])
```

An illustrative response (probabilities depend on the checkpoint):

```json
{
  "model": "Cloudflare/clef-flash",
  "answers": [
    {"type": "predicate", "name": "outage", "probability": 0.8},
    {"type": "choice", "name": "team", "choice": "technical", "confidence": 0.5,
     "probabilities": [{"value": "technical", "probability": 0.75}, {"value": "billing", "probability": 0.25}]},
    {"type": "score", "name": "urgency", "score": 1.6, "confidence": 0.4,
     "probabilities": [{"value": 0, "label": "Low", "probability": 0.1},
                       {"value": 1, "label": "Medium", "probability": 0.2},
                       {"value": 2, "label": "High", "probability": 0.7}]}
  ],
  "usage": {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 120
  }
}
```

Questions and answers preserve order. Names are optional (returned as `null`
when omitted); repeated names do not overwrite answers. Supply 1–200 questions,
2–255 distinct choice values, or 2–10 score levels, subject to model limits.
Choice values can be strings or Booleans: `true` and `"true"` remain distinct.
Each question requires `instructions`. Independent labels should use separate
predicates; `multi_label` is not part of this schema. A backend refusal is
represented as `{"type": "refusal", "name": ...}`; the adapter does not add
refusal behavior to models that lack it.

## Compatibility and local behavior

Input accepts text or ordered messages with user/developer/system/assistant
roles. User content supports `input_text` and `input_image` parts with HTTP(S)
or image data URLs (at most 128 images). Developer/system content is text;
assistant content can contain `output_text` or `refusal` parts. Function and
custom tool calls and matching outputs can appear as evidence. Calls are never
executed. Conversations are serialized into the model's native state with
roles, order, and image indices preserved; they do not use OpenAI's tokenizer
or instruction hierarchy.

Local extensions are top-level `images`, `videos`, and `audio`, media-only
requests, and an optional HTTP `model` when a decision model is preloaded.
The Python API also accepts model-supported PIL images, waveform arrays, and
file objects. Serialize arbitrary JSON evidence to text with `json.dumps`.
The native model method retains named questions, `criteria`, and `bool`/`noul`;
it is internal to the scheduler and the System One adapter.

`prompt_cache_key` partitions cache reuse within the server tenant;
`prompt_cache_options.mode` supports `implicit`. Cached and cache-write tokens
are reported by supporting engines and are zero otherwise. Cache retention is
bounded by the local LRU budget. Explicit cache breakpoints, retention TTLs,
image detail overrides (`low`/`high`/`original`), file inputs, and input item references
are rejected. `safety_identifier` is accepted as request metadata; it does not
activate an OpenAI abuse-monitoring service. HTTP responses include
`x-request-id` and the existing `x-clef-cached-tokens` header.

This is schema compatibility, not numerical parity with OpenAI. Probabilities
come from the selected local checkpoint after its calibration. Values are
rounded to four decimals, so displayed distributions can differ slightly from
one. Choice confidence is `(max_probability - 1/n) / (1 - 1/n)`, clipped to
`[0, 1]`. Score confidence is one minus the probability-weighted distance from
the modal level divided by the uniform distribution's distance from the
center, clipped at zero. These retain our System One statistics; OpenAI does
not document its confidence calculation. Native decision models generate no
answer tokens, so output and reasoning token usage are zero.

## CLI

Use `--input`, `--input-file`, `--questions`, `--questions-file`, or `--image`
for individual inputs. `--state` and `--state-file` remain text-input aliases.
`--input @input.json` reads an input array; JSON request files use only the new
`input`/question-array schema. Models that read speech also support `--audio`.

```sh
python -m mlx_vlm decide --request request.json
cat request.json | python -m mlx_vlm decide --request -
cat requests.jsonl | python -m mlx_vlm decide --request - --jsonl --batch-size 4
```

Each request contains `model`, `input`, and `questions`; supported media may
replace input text. `--model` overrides checkpoints in request files. JSONL
output preserves input order. Loader diagnostics go to stderr, leaving stdout
as JSON. Repeatable `--video` values contain JSON arrays of frame URLs.
`--images` accepts multiple references as an alternative to repeatable `--image`.

To compare checkpoints with the same requests:

```sh
python -m mlx_vlm decide --model ./model-4bit --request requests.jsonl --jsonl \
  --compare-model ./model-bf16 --report drift.json \
  --max-probability-drift 0.01 --max-choice-flips 0
```

The comparison uses unrounded distributions and matches option labels. A
failing gate exits with status 1. The candidate is released before the reference
loads; comparisons run with one request per batch to isolate checkpoint drift
from batch-dependent rounding. Models must expose unrounded distributions
through their decision engine to support comparison. This is a regression
check, not evidence of equal accuracy or calibration.

## Shared scheduling

Both HTTP endpoints use one bounded scheduler and the standard model loader:

```sh
python -m mlx_vlm server --decision-model Cloudflare/clef-flash \
  --decision-max-length 65536 --decision-batch-size 4 \
  --decision-prefill-step-size 512 --decision-prefix-cache-mb 256 \
  --decision-max-pending 64
```

The limits also have `MLX_VLM_DECISION_` environment variables: `MAX_LENGTH`,
`BATCH_SIZE`, `PREFILL_STEP_SIZE`, `PREFIX_CACHE_MB`, and `MAX_PENDING`. The CLI
accepts `--max-length`, `--batch-size`, `--prefill-step-size`, and
`--prefix-cache-mb`.

Models can implement `make_decision_engine(processor, **settings)`. The shared
scheduler only calls `prepare`, `step`, and `finish`; model-specific tensors,
encoding, and caches stay in the model backend. Models without this capability
continue using their native `predict` method. Their calls run serially inside
the worker, with cancellation checks between calls.

Clef's engine provides bounded chunked prefill, admission of new requests between
chunks, and byte-bounded LRU state-prefix caching. Cache entries include recurrent
state, KV state, and hidden states. Keys include media, tenant identity, and any `prompt_cache_key`.
Questions can change while the state prefix is reused. Image and video inputs
can share a request. The default complete input limit is the smaller of 65,536
and the model's context limit; overflow raises an error without truncating state.

A zero prefix-cache budget disables reuse and avoids an extra split at the
state/schema boundary. A batch size of one skips the batch coalescing delay.
Smaller chunks provide more cancellation opportunities at a possible throughput
cost. Cancellation is cooperative: it cannot interrupt an already-running GPU
kernel. Clef checks it between backbone layer groups and head layers; other
models need their own incremental engine for cancellation inside a forward pass.

A full queue returns HTTP 429. Disconnects cancel queued jobs and signal active
jobs. Shutdown drains the worker. Runtime metrics include decision queue depth
and active request count.

## TypeSafe System One compatibility

`POST /v1/systemone` retains the strict TypeSafe request and response schemas:
`state`, named questions with `criteria`, and named `noul`, `choice`, and `score`
answers. Confidence formulas, four-decimal rounding, zero output-token usage,
and the `x-typesafe-request-id`/`x-clef-cached-tokens` headers are preserved.
`model` and `state` are required. Images and video frames must be HTTP(S) URLs
or image data URLs on this endpoint.

```json
{
  "model": "Cloudflare/clef-flash",
  "state": "The checkout service is down",
  "questions": {
    "outage": {"type": "noul"},
    "team": {"type": "choice", "criteria": {"technical": "Bugs", "billing": null}},
    "urgency": {"type": "score", "criteria": ["Low", "Medium", "High"]}
  }
}
```

Send this body to `/v1/systemone`, or use
`mlx_vlm.decision.systemone(model, processor, request)`. The CLI takes the
OpenAI Decisions request shown above. Both endpoints share the same scheduler,
model loader, and raw probability handling.
