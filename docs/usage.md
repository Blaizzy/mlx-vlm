# Usage

## Command Line Interface (CLI)

Generate output from a model:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

## Chat UI with Gradio

Launch the chat interface:

```bash
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

## Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## Conversation compaction and APC

Compaction replaces older conversation turns with a model-generated handoff and
retains recent turns. Automatic Prefix Caching (APC) then matches the actual
rendered prompt. Only an unchanged prefix can reuse KV state: retained messages
after a new summary must be prefilled again because their old KV states depended
on the discarded history. Repeated continuations can cache the new summary and
retained turns normally. Compaction also works with APC disabled.

Start a server with APC:

```bash
APC_ENABLED=1 python -m mlx_vlm.server --model openbmb/MiniCPM5-2B --port 8080
```

There are two ways to manage compaction:

| Client | Integration |
| --- | --- |
| [Pi](https://github.com/earendil-works/pi/blob/0f8740bb65638180403a225ad7ec4d0cc1f8dedf/packages/coding-agent/docs/compaction.md) | Keep its session summary and retained-message boundary; send the rebuilt conversation through its normal provider. |
| [OpenCode](https://github.com/anomalyco/opencode/blob/0112a92c416f5ad833d96e7a8308441f0a875d94/packages/core/src/session/compaction.ts) | Keep its compaction/pruning policy and send summary plus recent messages normally. |
| [Hermes](https://github.com/NousResearch/hermes-agent/blob/aea969677c60a1bb72fe227fdfb98f196a2092cc/website/docs/developer-guide/context-compression-and-caching.md) | Keep its context compressor; native compaction through a local server needs a client adapter/context-engine integration. |
| [Codex / Responses protocol](https://developers.openai.com/api/docs/guides/compaction) | Call `/v1/responses/compact`, opt into automatic compaction, or send Codex's terminal `compaction_trigger` on `/v1/responses`. |

Client-owned summaries work through Chat Completions, Messages, or Responses
without a server-specific marker. Their lifecycle remains the client's
responsibility. The server never silently rewrites Chat Completions or Messages
history. These protocols have different client policies; see the tested CLI
versions and configurations below. Hermes has protocol coverage only.

For explicit server compaction, pass the **whole returned output array** into
the next request:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    default_headers={"X-APC-Tenant": "my-workspace"},
)
model = "openbmb/MiniCPM5-2B"
# history is the full Responses input array collected by your agent.
compacted = client.responses.compact(model=model, input=history)
response = client.responses.create(
    model=model,
    input=[*compacted.output, {"role": "user", "content": "Continue the task."}],
    max_output_tokens=1024,
    store=False,
)
```

`/responses/compact` is also available without the `/v1` prefix. It is a
non-streaming endpoint. Optional MLX extensions, passed with SDK `extra_body`,
are `max_output_tokens` (summary budget; default 1024, maximum 16384) and
`keep_tokens` (recent-context target; the latest complete user exchange always
stays). Resend the same `instructions` and `tools` on compaction and continuation
requests if they were used to render the original prompt. Generated handoffs
have assistant authority; system/developer messages keep their original roles.

For automatic compaction, add this field to a Responses request:

```json
{"context_management": [{"type": "compaction", "compact_threshold": 24000}]}
```

The server checks the rendered input token count **before generation**, including
for streaming requests. This implementation does not compact mid-generation.
Set the threshold below the context limit, leaving room for the summary prompt
and its output. When compaction happens, the response output starts with a
`type: "compaction"` item. Append the output normally, or keep only the newest
compaction item and everything after it. `previous_response_id` chaining also
works while that response is stored. `/v1/responses/input_tokens` counts the
decoded context, not the encrypted payload's string length. Normal response
usage counts the final inference; explicit compact usage counts the summary pass.

For Codex's native compaction flow, a single terminal input item
`{"type": "compaction_trigger"}` on `/v1/responses` requests compaction without
generating an answer. Both streaming and non-streaming responses contain exactly
one compaction item, including when the history is too short to shorten. Its
usage describes the summary pass. Identical system/developer messages resent
after a capsule replace their carried copies, preventing instruction growth
across repeated Codex compactions.

The server preserves the latest exchange and never cuts across outstanding tool
calls. It makes one summary attempt and accepts it only if it fits the available
budget and reduces removable history to at most 60% of its original token count.
Fixed instructions, tool schemas, and the protected tail are excluded from this
reduction target. Empty or truncated summaries and insufficient context headroom return an error without
replacing conversation state. A short conversation with nothing to summarize is
returned unchanged. One oversized user/tool exchange cannot be shortened by this
policy. Separate reasoning side-channel items follow the normal Responses
renderer; this is a textual handoff, not a compressed hidden model state.

Compaction items contain authenticated encrypted conversation state, not KV
tensors. They remain usable after cache eviction or server restart with the same
model name, tenant, and encryption key. The key is created with owner-only
permissions at `$MLX_VLM_CACHE_HOME/compaction.key` (default
`~/.cache/mlx-vlm/compaction.key`); set `MLX_VLM_COMPACTION_KEY_FILE` to use a
different path or share the key between server processes. Losing/changing the key
invalidates existing items. OpenAI-issued opaque compaction items are not
interchangeable with MLX-issued items. `store=False` avoids the Responses registry;
APC persistence is configured separately.

Run the opt-in HTTP integration test on Apple Silicon with model access and
`pytest`, `httpx`, and `openai` installed:

```bash
MLX_VLM_COMPACTION_TEST_MODEL=openbmb/MiniCPM5-2B \
  python -m pytest -s mlx_vlm/tests/test_compaction_model.py
```

The test exercises SDK compaction/replay, automatic streaming compaction,
corrections across repeated compaction, cold/warm APC, cache reset, server restart,
and client-authored summaries through Chat Completions and Messages. It writes
token counts, timings, and server logs in pytest's temporary directory. Its small
synthetic recall task is a regression check, not a general summary-quality benchmark.

### CLI validation at a 10,000-token context limit

Tested on 2026-10-01 with `openbmb/MiniCPM5-2B` on Apple Silicon, using the actual
CLIs and a server with a hard context limit (verified through `/health`):

```bash
MAX_KV_SIZE=10000 MLX_VLM_MAX_TOKENS=1024 \
  APC_ENABLED=1 APC_DISK_ENABLED=0 APC_NUM_BLOCKS=2048 \
  python -m mlx_vlm.server --model openbmb/MiniCPM5-2B --port 8769
```

Each run established a project name, deployment port, and protected filename,
added repetitive history until automatic compaction occurred, checked recall,
changed the port, and checked recall again after further compactions.

| Client | Turns | Automatic compactions | Largest inference prompt | Recall before/after correction |
| --- | ---: | ---: | ---: | --- |
| Pi 0.99.2 | 35 | 4 | 8,786 | Passed |
| OpenCode 1.18.34 | 32 | 5 | 8,857 | Passed |
| Codex 0.158.0-alpha.2.1, pasted history | 32 | 26 | 8,048 | Passed |
| Same Codex, tool history | 18 | 13 | 7,696 | Passed |

Both Codex workloads use native compaction in the mlx-vlm server. The first
grows history through pasted user messages; the second executes commands and
adds their results as tool output.

A controlled replay compared APC on and off using these same captured HTTP
requests on an Apple M5 Max with 128 GiB RAM. Each mode ran twice, with mode order
reversed on the second pass. Model warm-up preceded measurement; each workload
started with an empty prefix cache and disk caching disabled. Original request
bodies, including histories and compaction capsules, were held fixed.

| Workload | Requests per pass | APC off, total seconds | APC on, total seconds | Speedup | Median first text token, off → on | Output tokens/s, off → on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pi | 39 | 46.56 | 43.62 | 1.07× | 557 → 413 ms | 49.1 → 52.4 |
| OpenCode | 42 | 36.83 | 33.68 | 1.09× | 623 → 413 ms | 30.5 → 33.4 |
| Codex, pasted history | 58 | 124.21 | 107.82 | 1.15× | 809 → 513 ms | 59.2 → 69.1 |
| Codex, tool history | 45 | 69.39 | 50.31 | 1.38× | 816 → 456 ms | 41.7 → 58.0 |

Totals are means of two passes and include summarization, excluding model loading,
CLI startup, and tool execution. Output throughput divides generated tokens by
total HTTP wall time; it is not decode-only throughput. First-token measurements
exclude buffered compaction and tool-only responses without streamed deltas. All
736 timed requests succeeded, with identical input token counts between modes;
generated lengths differed by at most 1.41% per workload. APC-off cached token
counts were zero. Cold-start replay cache shares with APC were 69.3%, 72.4%,
87.7%, and 91.3%; these differ from the original live sessions below. This measures
server performance for fixed histories, not a new autonomous CLI session.

The Codex tool run executed 14 successful `cat` commands against synthetic log
files. Pi and OpenCode used pasted logs with tools disabled. All final runs had
zero HTTP errors. APC served 69.3%, 76.1%, 87.7%, and 93.0% of input tokens
respectively, including summarization requests. These are aggregate cache reuse
measurements, not speedup measurements or reuse of invalidated suffix KV states.

**Pi:** register the local provider in `models.json` with
`baseUrl: "http://127.0.0.1:8769/v1"`, `api: "openai-completions"`, and a dummy
API key. Set the model's `contextWindow` to `10000`, `maxTokens` to `1024`, and
`reasoning` to `false`. In `settings.json` use:

```json
{
  "compaction": {"enabled": true, "reserveTokens": 2000, "keepRecentTokens": 1000},
  "defaultThinkingLevel": "off"
}
```

**OpenCode:** use `@ai-sdk/openai-compatible` with the same local URL, a dummy
API key, and both `model` and `small_model` pointing to
`mlx/openbmb/MiniCPM5-2B`. Set these model limits and top-level compaction options:

```json
{
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:8769/v1", "apiKey": "local-test"},
      "models": {
        "openbmb/MiniCPM5-2B": {"limit": {"context": 10000, "input": 10000, "output": 1024}}
      }
    }
  },
  "model": "mlx/openbmb/MiniCPM5-2B",
  "small_model": "mlx/openbmb/MiniCPM5-2B",
  "compaction": {"auto": true, "prune": false, "reserved": 2000}
}
```

The explicit `input` limit matters in this OpenCode version: without it,
`reserved` does not reduce the usable input budget, and compaction can start too
late for a large incoming message.

**Codex:** the tested CLI selects native compaction when the custom provider's
display name is `OpenAI`. This is version-specific client behavior; requests
still go to the configured localhost URL. Use these settings through `-c`
overrides or an isolated configuration:

```toml
model_provider = "mlx"
model = "openbmb/MiniCPM5-2B"
model_context_window = 10000
model_auto_compact_token_limit = 7500
model_reasoning_effort = "none"
model_reasoning_summary = "none"
web_search = "disabled"
project_doc_max_bytes = 0

[model_providers.mlx]
name = "OpenAI"
base_url = "http://127.0.0.1:8769/v1"
wire_api = "responses"
requires_openai_auth = false

[features]
enable_request_compression = false
```

The tool run used a threshold of `8000`. Disable request compression explicitly
(`--disable enable_request_compression` also works); this server does not decode
Codex's zstd request bodies. Test runs also disabled apps, plugins, multi-agent,
and memories and used a read-only sandbox. Codex used fallback model metadata;
the context limit was explicitly overridden on both client and server.

At 10K, Codex's fixed instructions and tools occupy roughly 6K, so native
compaction is frequent and saves less of the total prompt. The default custom
provider path uses Codex's local text compactor, which retained large pasted user
logs and eventually overflowed in an earlier trial; it is not covered by the
successful native-compaction results. Model behavior also needs separate
evaluation: OpenCode sometimes returned extra continuation text, and MiniCPM
once emitted a float for an optional integer tool argument. The successful tool
run requested only the command argument. These tests establish recall and
continuation across compactions, not general coding or instruction-following
quality.

### CLI validation through a 64,000-token context limit

The same model, hardware, and CLI versions were tested at 16,000, 32,000, and
64,000 tokens on 2026-10-01. K denotes 1,000 tokens. Both server and client limits
were set explicitly, and the server's effective limit was checked through
`/health`. Each completed live workload crossed two automatic compactions,
checking the original facts after the first and a corrected port after the second.
The peak inputs below are smaller than the configured windows because compaction
starts before the window is full.

For these runs, set `MAX_KV_SIZE` to the selected limit and `APC_NUM_BLOCKS=8192`
(131,072 nominal pool tokens at block size 16); keep `MLX_VLM_MAX_TOKENS=1024`
and `APC_DISK_ENABLED=0`. The default 8 GiB cache byte budget and 450,000 pool-tensor
limit remained enabled. Pi used 20% of the window for `reserveTokens` and 10%
for `keepRecentTokens`. OpenCode used the full window for both `context` and
`input`, output limit 1024, and 20% for `reserved`. Codex used the native-provider
configuration above, `model_context_window` equal to the window, and
`model_auto_compact_token_limit` at 80%; its tool-output limit was 12,000 tokens.

Performance again uses two paired replays per APC mode, with mode order reversed
on the second pass. Every client/mode/pass starts a fresh server, warms the model,
and resets the prefix cache before measurement. This tighter isolation replaced
an initial shared-server 16K trial whose first OpenCode request repeatedly stalled;
the entire 16K comparison was rerun. The earlier 10K baseline used a smaller pool
and reset the cache between clients sharing a server. Workload lengths also vary
with the context window: compare APC modes within a row, not raw durations across
windows.

| Context limit | Client | Requests per pass | Peak input | APC off (s) | APC on (s) | Speedup | Median first text token, off → on |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | Pi | 19 | 13,876 | 28.01 | 21.34 | 1.31× | 1150 → 655 ms |
| 16K | OpenCode | 20 | 13,381 | 26.86 | 20.58 | 1.31× | 1024 → 657 ms |
| 16K | Codex, pasted history | 14 | 13,830 | 29.17 | 14.84 | 1.97× | 1780 → 704 ms |
| 16K | Codex, tool history | 20 | 12,355 | 40.59 | 20.32 | 2.00× | 1508 → 654 ms |
| 32K | Pi | 19 | 26,619 | 55.72 | 44.84 | 1.24× | 2192 → 1574 ms |
| 32K | OpenCode | 22 | 27,912 | 65.37 | 46.88 | 1.39× | 2259 → 1520 ms |
| 32K | Codex, pasted history | 18 | 28,696 | 80.41 | 36.38 | 2.21× | 3328 → 1825 ms |
| 32K | Codex, tool history | 27 | 25,398 | 97.79 | 38.93 | 2.51× | 2383 → 1687 ms |
| 64K | Pi | 19 | 52,159 | 162.99 | 164.95 | 0.99× | 7071 → 4563 ms |
| 64K | OpenCode | 22 | 53,297 | 167.13 | 161.90 | 1.03× | 5274 → 4106 ms |
| 64K | Codex, pasted history | 20 | 57,783 | 212.47 | 87.28 | 2.43× | 6911 → 4701 ms |
| 64K | Codex, tool history | Incomplete | 38,059 | — | — | — | — |

Times include all HTTP inference and summarization in a workload, excluding model
load, warm-up, CLI startup, and tool execution. Original request bodies, histories,
and compaction capsules are fixed during replay. All 880 timed requests in
this extension completed without HTTP errors; paired inputs and request hashes
matched, and APC-off cached token counts were zero. Generated lengths differed
by up to 6.56% per workload. These timings measure fixed-trace
server performance; quality was checked separately in the actual CLI sessions.
The small 64K Pi/OpenCode timing differences are below the observed spread
between their repeated runs; two passes do not establish a reliable speedup there.

Pi and OpenCode passed both recall checks at every tested window. Codex with pasted history
passed at 16K and 64K, but at 32K returned the workspace's project label instead
of `ORCHID`, despite the compaction handoff retaining `ORCHID`; the port correction
and protected filename survived. That wrong project answer also occurred in both
APC modes during replay. Codex tool history passed recall at 16K and 32K, but the
32K live run executed only 10 of 11 requested file reads: one response ended
without a tool call before the second compaction.

At 64K, Codex tool generation stalled after four successful reads, returning
EOS-only responses from 36,099 input tokens onward. A bounded run of 40 requested
log batches reached no compaction, so it has no completed-workload timing above.
Replaying the first 12 requests with APC off and on reproduced the same three
empty responses in both modes. This is a long-context continuation failure in
this model/backend combination, not evidence that APC caused it; the root cause
was not established.

The 64K Pi and OpenCode APC runs both filled 5,357 pool blocks, reaching the
default tensor limit despite the larger nominal block pool. After the first
compaction, growing prompts mostly reused only the fixed instruction prefix.
Their aggregate cached-input shares were 31.0%
and 33.2%, compared with
80.4% for Codex with pasted history. The store
path skips new blocks at that tensor limit before trying pool eviction, which
limits admission of the new compacted history. This benchmark retains that
configuration and exposes the limitation; increasing the configured context
alone does not ensure useful cache reuse after compaction.

## MoE Offloading

Run a mixture-of-experts checkpoint that is larger than available RAM by paging routed experts from disk. Repack the checkpoint into an offloaded store once, then load it as usual — `load()` detects the offloaded layout automatically:

```bash
python -m mlx_vlm moe_offload --build /path/to/checkpoint --out /path/to/offloaded
python -m mlx_vlm.generate --model /path/to/offloaded --prompt "Explain how photosynthesis works." --max-tokens 100
```

Serving works the same way. Bound the resident expert set with `--expert-cache-gb`, and inspect cache hits/misses/evictions at `/v1/moe-offload/stats`:

```bash
python -m mlx_vlm.server --model /path/to/offloaded --expert-cache-gb 8
```

## Speculative Decoding

Speed up generation 2–3× using a lightweight drafter model that predicts multiple tokens per round, verified in parallel by the target model.

### CLI

```bash
python -m mlx_vlm.generate \
    --model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --prompt "Write a quicksort in Python." \
    --max-tokens 512 --temperature 0 --enable-thinking
```

Liquid AI's LFM2.5 DSpark drafter is also auto-detected:

```bash
python -m mlx_vlm.generate \
    --model LiquidAI/LFM2.5-2.6B \
    --draft-model LiquidAI/LFM2.5-2.6B-DSpark \
    --prompt "Write a concise note about speculative decoding." \
    --max-tokens 256 --temperature 0
```

DSpark decoding currently supports greedy sampling (`temperature=0`).

EAGLE-3 speculators are also supported and auto-detected from Speculators configs:

```bash
python -m mlx_vlm.generate \
    --model mlx-community/gemma-4-31B-it-bf16 \
    --draft-model RedHatAI/gemma-4-31B-it-speculator.eagle3 \
    --prompt "Write a concise note about speculative decoding." \
    --max-tokens 256 --temperature 0
```

MiniMax M3 uses the same `eagle3` path with the released
`Inferact/MiniMax-M3-EAGLE3` drafter:

```bash
python -m mlx_vlm.generate \
    --model ~/MiniMax-M3-4bit \
    --draft-model ~/MiniMax-M3-EAGLE3 \
    --draft-kind eagle3 \
    --draft-block-size 3 \
    --prompt "Write a concise note about MiniMax Sparse Attention." \
    --max-tokens 256 --temperature 0
```

The public MiniMax M3 BF16 checkpoint advertises MTP metadata but does not
publish `mtp` or `nextn` tensors; use `Inferact/MiniMax-M3-EAGLE3` for that
checkpoint.

Works with images too:

```bash
python -m mlx_vlm.generate \
    --model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --image examples/images/cats.jpg \
    --prompt "Describe this image." \
    --max-tokens 256 --temperature 0 --enable-thinking
```

### Python — Single Sequence

```python
from mlx_vlm import load
from mlx_vlm.generate import stream_generate
from mlx_vlm.speculative.drafters import load_drafter

model, processor = load("Qwen/Qwen3.5-4B")
drafter = load_drafter("z-lab/Qwen3.5-4B-DFlash")

for result in stream_generate(
    model, processor,
    prompt="Write a quicksort in Python.",
    max_tokens=512,
    temperature=0,
    draft_model=drafter,
    enable_thinking=True,
):
    print(result.text, end="", flush=True)

# Acceptance stats
print(f"\nAccepted {sum(drafter.accept_lens)/len(drafter.accept_lens):.1f} tokens/round")
```

### Python — Batch Generate

Process multiple prompts in parallel:

```python
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate import (
    _dflash_rounds_batch,
    _make_cache,
    generation_stream,
)
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.sample_utils import make_sampler

model, processor = load("Qwen/Qwen3.5-4B")
drafter = load_drafter("z-lab/Qwen3.5-4B-DFlash")
tok = processor.tokenizer
lm = model.language_model
sampler = make_sampler(temp=0)
eos_id = tok.eos_token_id

prompts = [
    "Write a quicksort in Python.",
    "What is the capital of France?",
    "Explain hash tables in 3 sentences.",
]

# Tokenize and left-pad to uniform length
texts = [
    apply_chat_template(
        processor, model.config, p,
        num_images=0, num_audios=0, enable_thinking=True,
    )
    for p in prompts
]
encoded = [tok.encode(t) for t in texts]
max_len = max(len(e) for e in encoded)
padded = [[0] * (max_len - len(e)) + e for e in encoded]
input_ids = mx.array(padded, dtype=mx.int32)
B = len(prompts)

# Create batch-aware caches and prefill
prompt_cache = _make_cache(lm, [0] * B)
lm._position_ids = None
lm._rope_deltas = None

target_layer_ids = list(drafter.config.target_layer_ids)
out = lm(input_ids, cache=prompt_cache, capture_layer_ids=target_layer_ids)
hidden = mx.concatenate(out.hidden_states, axis=-1)
first_bonus = sampler(out.logits[:, -1:]).squeeze(-1)
mx.eval(first_bonus, hidden, out.logits)

# Generate — finished sequences are automatically removed from
# the batch and the drafter restarts for the new batch size.
tokens_per_seq = [[] for _ in range(B)]
for tok_list, _ in _dflash_rounds_batch(
    model, drafter, prompt_cache, hidden,
    first_bonus=first_bonus,
    max_tokens=256,
    sampler=sampler,
    token_dtype=mx.int32,
    stop_check=lambda seq_idx, token_id: token_id == eos_id,
):
    for i, t in enumerate(tok_list):
        if t is not None:
            tokens_per_seq[i].append(t)

# Decode results
for i in range(B):
    all_toks = [int(first_bonus[i].item())] + tokens_per_seq[i]
    print(f"--- {prompts[i]}")
    print(tok.decode(all_toks))
```

### Supported Models

| Target | Drafter | Notes |
|--------|---------|-------|
| `Qwen/Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` | Text + image. ~2.5× speedup on code/reasoning. |
| `LiquidAI/LFM2.5-2.6B` | `LiquidAI/LFM2.5-2.6B-DSpark` | Text. Nine Markov-corrected proposals with exact LFM2 target verification. |
| `meta-models/Muse-Glimmer-30B` | `meta-models/Muse-Glimmer-30B-assistant` | Text + image. Native 5-layer, 16-token DFlash assistant. |
| `MiniMaxAI/MiniMax-M3` | `Inferact/MiniMax-M3-EAGLE3` | Text, image, and video target. Uses `--draft-kind eagle3`. |

The drafter is loaded via the shared `load_model` path. DFlash checkpoints are
detected from `dflash_config` or the `muse_glimmer_assistant` model type;
EAGLE-3 checkpoints are detected from
`speculators_model_type` or EAGLE-3 architecture metadata. Native MTP sidecars
for supported model families are detected from their `model_type`.

## Server (FastAPI)

```bash
python -m mlx_vlm.server
```

See `README.md` for a complete `curl` example.

### Live settings (`/v1/settings`)

Read and change a curated set of server settings at runtime, without a
restart. `GET` lists the settings the server accepts; `PATCH` changes them.

```bash
# list the available settings and their current values
curl http://127.0.0.1:8080/v1/settings

# merge: only the settings you list are changed
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"kv_quant_scheme": "turboquant"}'

# replace: reset everything to its boot-time default, then apply these
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"op": "replace", "values": {"apc_enabled": true}}'
```

Changes take effect on the next request. Most settings reload the affected
model first — KV, APC, and speculative-decoding settings reload text models,
`vision_cache_size` reloads image models — while `max_kv_size` and
`token_queue_timeout` apply to new requests without a reload.

The response reports which settings were applied and which were rejected;
unknown names and invalid values are rejected and never applied.

## Distributed Inference

mlx-vlm supports distributed inference across multiple computers. It works by sharding the language model (not the vision tower), because the LLM is much larger and vision embeddings only need to be computed once.

The parallel implementation is compatible with mlx-lm sharding primitives.

The following command shows how you can run Kimi K2.6, a 1T parameter model, on several computers. For a smaller option, you can try `[mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16)`.

```bash
mlx.launch \
    --hostfile ring-thunderbolt.json \
    --backend jaccl \
    --hostfile /path/to/hosts.json \
    --env MLX_METAL_FAST_SYNCH=1 \
    -- \
    mlx-vlm/examples/sharded_generate.py \
    --model moonshotai/Kimi-K2.6 \
    --prompt "Describe this image" \
    --image mx-vlm/examples/images/scene_1.jpg
```

We recommend you use the JACCL protocol over Thunderbolt. For more information, please refer to [the MLX distributed communication guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).
