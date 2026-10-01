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
history. These protocols have different client policies. Hermes has protocol
coverage only.

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
