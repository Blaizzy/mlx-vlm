# Clef and Clef-Flash

Native MLX support for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
(9B) and [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (27B).
Both use the existing Qwen3.5 text/vision backbone plus the released joint schema
head. All questions share one backbone forward pass. There is no text decoding.

## Run

```bash
python -m mlx_vlm server --decision-model Cloudflare/clef-flash --port 8080
```

```bash
curl http://localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Cloudflare/clef-flash",
    "state": "Our checkout is down and customers cannot place orders.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Invoices", "technical": "Bugs or outages"}
      },
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}
    }
  }'
```

The CLI accepts files, stdin, inline arguments, and JSONL batches:

```bash
python -m mlx_vlm decision --request request.json
cat request.json | python -m mlx_vlm decision --request -
python -m mlx_vlm decision --model Cloudflare/clef-flash \
  --state 'Checkout is down' --questions '{"outage":{"type":"noul"}}'
python -m mlx_vlm decision --request requests.jsonl --jsonl --batch-size 4
```

`--questions @questions.json` and `--state @state.json` read structured JSON from
files. `--images URL...` and repeatable `--video '["FRAME_URL", "FRAME_URL"]'`
add media to inline requests. `--model` overrides the model in input files.
JSONL results preserve input order; stdout contains only response JSON and
loader diagnostics go to stderr. JSONL processing has a bounded submission
window and fails on invalid input. Ctrl-C cancels queued and active work.
For Python use, load with `mlx_vlm.utils.load_model` / `load_processor` and call
`mlx_vlm.decision.systemone(model, processor, request)`.

The original checkpoints are detected by `joint_head_config.json`. Loading also
includes `joint_head.safetensors`, which is absent from the backbone shard index.
Converted checkpoints embed the head configuration in `config.json` and include
head weights in the standard model shards. Remote Python code is not executed.

```bash
python -m mlx_vlm convert --hf-path Cloudflare/clef-flash \
  --mlx-path ./clef-flash-4bit -q --q-bits 4
python -m mlx_vlm server --decision-model ./clef-flash-4bit
```

Use the same local path in the request's `model`. Conversion keeps the joint
head **and its lexical output embeddings** in floating point by default, including
with mixed-bit recipes. With tied weights, the shared input/output embedding
matrix is protected. `--decision-quantization backbone` permits quantizing the
lexical embeddings for additional memory savings while still protecting the head.
Quantized lookups gather and dequantize only the required vocabulary rows.
Choose `--q-bits 8` for a less aggressive weight reduction, or omit `-q` to retain
floating-point weights. No quantization policy guarantees unchanged decisions.

Compare a candidate against a floating-point reference on representative requests:

```bash
python -m mlx_vlm decision --request evaluation.jsonl --jsonl \
  --model ./clef-flash-4bit --compare-model Cloudflare/clef-flash \
  --report drift.json --max-probability-drift 0.01 --max-choice-flips 0
```

The report uses **unrounded** probabilities, measures maximum/mean absolute drift
and counts argmax changes across all question types. Exit status 1 means a
threshold failed. Comparison runs both checkpoints with batch size one and the
same prefill step size to isolate weight changes from batch-dependent rounding.
The reference loads after the candidate is released. Comparison
mode retains evaluation inputs and distributions in host memory; ordinary JSONL
inference streams them. This is a regression check against a reference, not an
accuracy or calibration guarantee. Use an unchanged media source for comparisons.

## Server contract

`POST /v1/systemone` accepts `model`, `state`, and a nonempty map of `questions`.
`state`, question `instructions`, and descriptions can be text, JSON objects, or
arrays. Omitted instructions use the question ID, matching Clef's trained format.

| Question | Criteria | Answer |
| --- | --- | --- |
| `noul` | Optional `true`/`false` descriptions | `type`, `noul` = P(true) |
| `choice` | 1–255 named descriptions, including null | `type`, `choice`, `confidence`, `probabilities` |
| `score` | 1–10 ordered descriptions | `type`, expected `score`, `confidence`, `legend`, `probabilities` |

Responses contain `model`, `answers` keyed by question ID, and
`usage: {input_tokens, output_tokens}`. Input counts the whole encoded prompt
once; output tokens are zero. Scores use zero-based levels. Probabilities are
softmax-normalized per question and rounded to four decimal places.

Confidence follows the published
[TypeSafe adapter formulas](https://github.com/typesafe-ai/system-one-adapter-python/blob/main/src/system_one_adapter/_utils/confidence_metrics.py):
choice confidence rescales the peak probability from uniform to one; score
confidence measures distance from the modal level relative to a uniform rubric.
For a single option confidence is one. These statistics are not empirical
guarantees of correctness. Cloudflare's sample `systemone` helper instead uses
the maximum probability for both kinds; this port intentionally differs there.

Optional `images` contains up to 10 HTTP(S) or image data URLs. Optional `videos`
contains up to 4 lists of frame URLs, 1–64 frames each, with matching dimensions
within each video. Images and videos can coexist in one request; each modality
uses its own vision features and shares the multimodal position map. The Python
encoder also accepts PIL images / NumPy video arrays.

Up to 64 questions are accepted. The complete-input budget defaults to 65,536
tokens, bounded by the checkpoint's backbone position limit. Set
`--decision-max-length` on the server or `--max-length` on the decision CLI to
change it. Without a decision-specific server setting, an explicit
`--max-kv-size` is honored. Counts include the state, schema, special tokens, and
media tokens. Excess input is rejected without truncation. Raising the budget
increases memory and latency; accepting a longer input does not validate the
model's decision quality at that length. Invalid schemas return 422; unsupported models return 400; missing
repositories return 404. Unknown fields, including generation controls, are
rejected. The existing server API key applies. Responses include
`x-typesafe-request-id` and `x-clef-cached-tokens`; `/openapi.json` describes the
schema. A saturated queue returns 429. Shutdown cancels pending and active work.

`GET /v1/models` retains the server's OpenAI catalog format, as SGLang does.
TypeSafe's `models.list()` catalog shape is therefore not supported. Requests must
name an actual model ID/path; `jev-latest` is not mapped to a different model.

## API design comparison (checked 2026-10-09)

| Reference | Serving approach | Contract implication |
| --- | --- | --- |
| [TypeSafe Jev](https://api.typesafe.ai/redoc) | Hosted decision model; `noul`, `choice`, `score` | Use its request and answer shapes as the public contract. |
| [SGLang](https://github.com/sgl-project/sglang/blob/main/docs/docs/supported-models/decision_models.mdx) | Generic label scoring plus model-specific native decision paths, including Clef's joint head | Follow its distinction between label-token scoring and native heads. Native heads have no vocabulary `label_mass`. |
| [vLLM structured decisions](https://docs.vllm.ai/en/latest/serving/online_serving/structured_decisions/) | The documented endpoint scores first-token labels per question and supports `choice` | Its `structured_decision` envelope, diagnostics, and vocabulary-based confidence differ from Jev; they are not substituted for Clef's head. |
| MLX-VLM | Native Clef head over one shared prefill | TypeSafe answer shapes and confidence formulas; zero generated tokens. |

These APIs and backends are evolving. SGLang's model card launch example uses
the `dev-clef` image; support should be checked against the deployed version.

## Tradeoffs and limitations

- Both still execute a large backbone prefill. Eliminating decoding does not
  eliminate input-length costs, weight memory, or image preprocessing. Published
  H200 latency is not an Apple Silicon benchmark.
- The head attends to the full prompt and couples the questions. Adding or
  rewording questions/options can alter probabilities. A supplied choice set
  can omit the right answer; applications should include an abstention option
  when needed and evaluate thresholds on representative data.
- Cloudflare's published evaluation shows substantial task dependence: Flash
  is much weaker than Clef on RAGTruth and CLINC150+OOS. Neither model dominates
  Jev on every benchmark. Treat the smaller model as a workload-specific choice.
- Continuous scheduling batches backbone prefill for requests at equal token
  offsets and admits new requests between chunks. The joint head runs separately
  for each schema. Different offsets, lengths, and model switches can reduce
  batching efficiency. This is not a measured throughput claim.
  Chunking and batching can also introduce floating-point differences, especially
  with BF16. Evaluate close decisions under the intended serving settings.
- Prefix reuse caches the exact state/media prefix, including both KV/recurrent
  caches and final hidden states. Different questions can reuse that prefix;
  changed state tokens or decoded media cannot. The cache is a byte-bounded LRU,
  scoped to the loaded model and the existing `X-APC-Tenant` / `X-Tenant-Id` salt.
  Model switches, unload, and shutdown discard it. An entry larger than the
  budget is not retained. The budget excludes active requests and model weights.
- Disconnects and Ctrl-C stop queued work immediately, and active work at
  cooperative checkpoints (every four backbone layers, each prefill chunk, and
  head layers). Already executing GPU operations and media I/O cannot be
  interrupted; a cancelled row sharing a batch is removed at the next chunk.
- KV quantization, streaming/chat generation, adapters, and generic LLM decision
  fallbacks are not implemented here.

Server scheduling controls:

```bash
python -m mlx_vlm server --decision-model ./clef-flash-4bit \
  --decision-max-length 65536 --decision-batch-size 4 \
  --decision-prefill-step-size 512 --decision-prefix-cache-mb 256 \
  --decision-max-pending 64
```

The corresponding environment variables are `MLX_VLM_DECISION_MAX_LENGTH`,
`MLX_VLM_DECISION_BATCH_SIZE`, `MLX_VLM_DECISION_PREFILL_STEP_SIZE`,
`MLX_VLM_DECISION_PREFIX_CACHE_MB`, and `MLX_VLM_DECISION_MAX_PENDING`.
The CLI exposes `--batch-size`, `--prefill-step-size`, and `--prefix-cache-mb`.
A zero cache budget disables reuse. Smaller batches/chunks reduce working memory
and scheduling delay, at a potential throughput cost. The server runtime snapshot
includes decision queue depth and active request count.

## Validation sources

The encoder and head are adapted under Apache-2.0 from Cloudflare's releases:
Clef-Flash revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` and Clef revision
`0b331204bb13fbd2ca93a64df1956f5b55478ce5`.
Shared decision contracts cover output shapes, finite logits, checkpoint
round-trips, quantization protection, and vocabulary row lookup. Model dimensions,
protected parameter paths, and optional media inputs live in `model_cases.json`;
Python tests select cases by capability. Protocol, CLI, scheduling, caching, and
cancellation checks are grouped in the existing subsystem suites. HTTP and CLI
formatting tests use deterministic responses without running a model.

Local verification on Apple M3 Ultra with MLX 0.32.3 / Transformers 5.19.0:

- 586 tests passed across processor, CLI, generation, server/audio, and shared
  model-contract coverage (2 skipped; 7 subtests passed).
- Released Flash head weights agreed with the upstream FP32 head within
  `1.1e-6` maximum absolute logit error on synthetic hidden states.
- A small complete Qwen3.5-plus-head model agreed within `9e-8` on decision
  logits after the normalization fix.
- All released tensor names/shapes matched strict loading: 882 for Flash and
  1,306 for Clef, including their heads.
- Both complete release checkpoints loaded and ran the text decision benchmark
  below on 2026-10-10. All weight files passed safetensors length/header checks.
  Full-checkpoint multimodal accuracy and quantized-model calibration have not
  been established.

### Local throughput (2026-10-10)

Apple M3 Ultra, 80 GPU cores, 512 GiB unified memory; original BF16 weights;
1,024 total input tokens and three questions per request (choice, score, noul).
The in-process server scheduler used a 512-token prefill step and one warmup round
followed by three timed rounds per case. Model loading and HTTP/network overhead
are excluded. Each answered question counts as one decision.

| Workload | Flash requests/s | Flash decisions/s | Clef requests/s | Clef decisions/s |
| --- | ---: | ---: | ---: | ---: |
| One request at a time, prefix cache disabled | 1.27 | 3.82 | 0.35 | 1.04 |
| Four concurrent requests, prefix cache disabled | 1.47 | 4.40 | 0.42 | 1.25 |
| One request at a time, warm state prefix | 4.18 | 12.54 | 1.21 | 3.62 |

The warm-prefix case reused 765 tokens with a 256 MiB cache budget. Single-request
median elapsed time was 786 ms for Flash and 2,865 ms for Clef without prefix
reuse, and 239 ms / 823 ms with reuse. These are short synthetic-text measurements,
not a sustained HTTP load test or a prediction for other input lengths, question
counts, media, or quantization settings.

### Lower-latency decisions

When prefix caching is disabled, prefill no longer splits at the state/schema
boundary solely to create a prefix that will never be stored. Requests shorter
than the configured prefill step can traverse the backbone once. A scheduler
batch size of one also skips the coalescing delay. Prefill remains bounded and
retains the cancellation checkpoints. Prefix caching and batch sizes above one
keep their existing behavior.
