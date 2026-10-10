# Decision serving and batch evaluation

`mlx_vlm.predict(model, processor, state, questions)`, `mlx_vlm.decide`, and
`/v1/decisions` share question validation and preserve each model's native scoring.
Supported question types and media are declared by the model. Clef, Decider,
Laya, and the LiquidAI d1 models use this interface.

## CLI

Existing `--state`, `--state-file`, `--questions`, `--questions-file`, `--image`,
and `--audio` inputs continue to work. Requests can also be supplied as JSON:

```sh
python -m mlx_vlm decide --request request.json
cat request.json | python -m mlx_vlm decide --request -
cat requests.jsonl | python -m mlx_vlm decide --request - --jsonl --batch-size 4
```

Each request contains `model`, `state`, and `questions`; media may replace the
state for models that support it. `--model` overrides the checkpoint in request
files. JSONL output preserves input order. Loader diagnostics go to stderr, so
stdout remains JSON. Models that read videos accept repeatable `--video` values,
each containing a JSON array of frame URLs. `--images` accepts multiple image
references as an alternative to repeatable `--image`.

To compare checkpoints with the same requests:

```sh
python -m mlx_vlm decide --model ./model-4bit --request requests.jsonl --jsonl \
  --compare-model ./model-bf16 --report drift.json \
  --max-probability-drift 0.01 --max-choice-flips 0
```

The comparison uses unrounded probability distributions and matches option
labels. A failing gate exits with status 1. The candidate is released before the
reference loads; comparisons run with one request per batch to isolate checkpoint
drift from batch-dependent rounding. Models must expose unrounded distributions
through their decision engine to support comparison. A comparison is a regression
check, not evidence that a model is calibrated or equally accurate.

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
state, KV state, and hidden states. Keys include media and tenant identity.
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

`/v1/systemone` adapts shared native predictions to the TypeSafe wire format.
It retains `noul`, `choice`, and `score`, their confidence formulas, four-decimal
output rounding, zero output-token usage, and the `x-typesafe-request-id` and
`x-clef-cached-tokens` headers. `model` and `state` are required in this format.
Image and video-frame references must be HTTP(S) URLs or image data URLs.

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

Use `mlx_vlm.decide --format systemone` for this schema. The previous
`mlx_vlm.decision` executable, `python -m mlx_vlm decision`, and Python
`mlx_vlm.decision.systemone(...)` helper remain compatibility entry points.
The adapter uses unrounded distributions when the backend exposes them;
otherwise it uses the model's native probabilities. Native `/v1/decisions`
answers and model-specific calibration remain unchanged.
