# Decider-2b decisions on MLX

Prepared checkpoint: [nativ-community/decider-2b](https://huggingface.co/nativ-community/decider-2b).

```python
from mlx_vlm import load, predict

model, processor = load("nativ-community/decider-2b")
result = predict(model, processor, "Please refund my duplicate charge", {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": ["billing", "technical", "sales"],
    },
})
print(result["answers"]["department"]["value"])
```

Use a checkpoint with `model_type: "decider2"` in its root `config.json`,
alongside the Qwen3.5 configuration and a `decision_config` object containing
the published `decider_config.json` settings. Tokenizer files belong at the
checkpoint root. This metadata must be prepared before loading the original
Mapika checkpoint or an older conversion labeled `qwen3_5_text`.

Weights use the existing Qwen3.5 text backbone and sanitization. No special
loader or runtime sidecar detection is required.

The shared `predict(model, processor, state, questions)` API returns named
answers with `type` and `value`. Decider supports `choice`, `score`, and `bool`;
`noul` is a Boolean alias. Choice returns a label and score returns an expected
level index, both with `probabilities`. Boolean answers include the probability
of true. Confidence, certainty, and isolated-level metrics remain in `metadata`.
Unsupported question types fail before inference. No answer tokens are generated.

Choice criteria are 2–255 labels or a mapping of labels to descriptions. Score
criteria are an ordered list of 2–10 level descriptions. Boolean criteria are
optional descriptions keyed by `false` and `true`.

`independent=True` evaluates questions separately. Score levels are isolated
according to the checkpoint settings unless overridden with `isolated`.
`independent=False` packs questions together, allowing later questions to attend
to earlier ones. Per-type temperatures use checkpoint settings with the global
temperature as fallback; isolated score rows use the score temperature.

`max_state_tokens` defaults to 32768. Inference performs a full forward pass,
so long inputs may exceed device memory.

## Command line

```sh
python -m mlx_vlm.decide --model nativ-community/decider-2b \
  --state "Please refund my duplicate charge" \
  --questions '{"department":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":["billing","technical","sales"]}}'
```

Use `--state-file request.txt` for UTF-8 text and `--questions-file questions.json`
for a JSON mapping of named questions. The command prints the same JSON result
as `predict()`, preserving each model's supported question types and scoring.
After installation, `mlx_vlm.decide` runs the same command.

## HTTP server

Start the existing server with `python -m mlx_vlm.server`, then send a request:

```sh
curl http://localhost:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nativ-community/decider-2b","state":"Please refund my duplicate charge","questions":{"department":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":["billing","technical","sales"]}}}'
```

`POST /v1/decisions` returns the shared prediction result directly and uses the
server's existing API-key authentication. Decision models have their own cache
entry and do not use token-generation workers. Requests are non-streaming and
preserve each model's supported question types and native scoring.
