# Speculative Decoding

Speed up generation by drafting several candidate tokens with a small "drafter" model and verifying them in a single target forward pass. Three drafter families are supported.

| Flag | Description |
|------|-------------|
| `--draft-model` | HuggingFace repo or local path for the drafter |
| `--draft-kind` | Drafter family — `dflash` (default), `eagle3`, or `mtp` (native/assistant MTP) |
| `--draft-block-size` | Override the drafter's configured block size |

See [Python API](#python-api) below for streaming, acceptance stats, and batch generation.

## DFlash, DFlash2, and DSpark

A lightweight block-diffusion drafter that predicts multiple tokens per round, typically 2–3× faster.

```sh
# Text generation with speculative decoding
mlx_vlm.generate --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0 --enable-thinking

# Also works with images
mlx_vlm.generate --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --image examples/images/cats.jpg \
  --prompt "Describe this image." \
  --max-tokens 256 --temperature 0 --enable-thinking

# Server with speculative decoding
mlx_vlm.server --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash
```

DFlash2 adds dynamic convolutions and a candidate-path selector. The published
Qwen3.8-27B checkpoint is auto-detected and uses the shared exact DFlash target
verification path. For the fastest quantized setup, convert the drafter to
4-bit; the verifier adapts between three and five rows from recent acceptance:

```sh
mlx_vlm.convert --hf-path z-lab/Qwen3.8-27B-DFlash2 \
  --mlx-path Qwen3.8-27B-DFlash2-4bit \
  --quantize --q-bits 4 --q-group-size 64

mlx_vlm.generate --model mlx-community/Qwen3.8-27B-4bit \
  --draft-model Qwen3.8-27B-DFlash2-4bit \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0

mlx_vlm.server --model mlx-community/Qwen3.8-27B-4bit \
  --draft-model Qwen3.8-27B-DFlash2-4bit
```

Liquid AI's DSpark checkpoint uses a Qwen3-style block drafter plus a learned
Markov correction head. It is auto-detected and runs through the exact target
verification path:

```sh
mlx_vlm.generate --model LiquidAI/LFM2.5-2.6B \
  --draft-model LiquidAI/LFM2.5-2.6B-DSpark \
  --prompt "Explain speculative decoding in three sentences." \
  --max-tokens 256 --temperature 0

mlx_vlm.server --model LiquidAI/LFM2.5-2.6B \
  --draft-model LiquidAI/LFM2.5-2.6B-DSpark
```

The published DSpark `block_size: 9` means nine proposals, or ten target rows
after adding the anchor token. On MLX, DSpark verifies seven proposals plus the
anchor by default: eight rows exactly fill the verifier threadgroup, while nine
or ten rows pad to sixteen and run slower. The trained width remains available
with `--draft-block-size 10`. The checkpoint's confidence head is loaded for
parity, and DSpark decoding currently requires greedy sampling
(`temperature=0`).

Muse Glimmer's published assistant checkpoint is auto-detected as DFlash:

```sh
mlx_vlm.generate --model meta-models/Muse-Glimmer-30B \
  --draft-model meta-models/Muse-Glimmer-30B-assistant \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0

mlx_vlm.server --model meta-models/Muse-Glimmer-30B \
  --draft-model meta-models/Muse-Glimmer-30B-assistant
```

DFlash draft-cache windowing is available from the Python API. During
speculative decoding the target model still verifies every proposed token with
its full KV cache; this knob only changes the DFlash drafter cache. When
`draft_window_size` is set, the drafter keeps at most that many recent committed
tokens in its own KV cache instead of attending over the full generated prefix.
That reduces draft-side cache length and memory, but it can lower acceptance
because the drafter has less context than the target verifier. On MLX, the full
draft cache is usually faster for Qwen3.5 DFlash, so windowing defaults to
`None`; set it only when you want to experiment with this compact recent-token
cache tradeoff:

```python
from mlx_vlm import load
from mlx_vlm.generate import generate
from mlx_vlm.speculative.drafters import load_drafter

model, processor = load("Qwen/Qwen3.5-4B")
draft_model, draft_kind = load_drafter("z-lab/Qwen3.5-4B-DFlash")
draft_model.config.draft_window_size = 256  # None disables windowing

result = generate(
    model,
    processor,
    "Write a quicksort in Python.",
    max_tokens=512,
    temperature=0,
    draft_model=draft_model,
    draft_kind=draft_kind,
)
```

## Gemma 4 MTP

[Multi-Token Prediction](https://ai.google.dev/gemma/docs/mtp/mtp): Google's 4-layer "assistant" drafter that shares K/V with the target and drafts multiple tokens autoregressively from a constant position. Pass `--draft-kind mtp` to dispatch the MTP round-loop.

```sh
mlx_vlm.generate --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
  --draft-kind mtp --draft-block-size 4 \
  --prompt "Explain speculative decoding in 3 sentences." \
  --max-tokens 256 --temperature 0

# Server
mlx_vlm.server --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
  --draft-kind mtp --draft-block-size 4
```

Supported pairings (target ↔ drafter):

| Target                          | Drafter                                  |
|---------------------------------|------------------------------------------|
| `mlx-community/gemma-4-E2B-it-bf16`         | `mlx-community/gemma-4-E2B-it-assistant-bf16`        |
| `mlx-community/gemma-4-E4B-it-bf16`         | `mlx-community/gemma-4-E4B-it-assistant-bf16`        |
| `mlx-community/gemma-4-26B-A4B-it-bf16`     | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`    |
| `mlx-community/gemma-4-31B-it-bf16`         | `mlx-community/gemma-4-31B-it-assistant-bf16`        |

Measured speedups (greedy, byte-identical output): up to **3.94×** on 26B-A4B and **2.29×** on 31B at B=4. See [`mlx_vlm/speculative/drafters/gemma4_assistant/README.md`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/gemma4_assistant/README.md) for full sweeps and architecture notes.

## Gemma 4 EAGLE-3

[EAGLE-3](https://sgl-project.github.io/SpecForge/concepts/EAGLE3.html) drafts from three target hidden-state captures with a lightweight one-layer speculator. The Red Hat Speculators checkpoint auto-detects as `--draft-kind eagle3`.

```sh
mlx_vlm.generate --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model RedHatAI/gemma-4-31B-it-speculator.eagle3 \
  --prompt "Explain speculative decoding in 3 sentences." \
  --max-tokens 256 --temperature 0

# Server
mlx_vlm.server --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model RedHatAI/gemma-4-31B-it-speculator.eagle3
```

## MiniMax M3 EAGLE-3

MiniMax M3 supports the released `Inferact/MiniMax-M3-EAGLE3` drafter. Convert
the target with `mlx_vlm.convert` because `mlx_lm.convert` does not know the
`minimax_m3_vl` model type.

```sh
mlx_vlm.convert \
  --hf-path MiniMaxAI/MiniMax-M3 \
  --mlx-path ~/MiniMax-M3-4bit \
  --quantize --q-bits 4 \
  --trust-remote-code

mlx_vlm.convert \
  --hf-path Inferact/MiniMax-M3-EAGLE3 \
  --mlx-path ~/MiniMax-M3-EAGLE3

mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --draft-model ~/MiniMax-M3-EAGLE3 \
  --draft-kind eagle3 \
  --draft-block-size 3 \
  --prompt "Explain MiniMax Sparse Attention in one paragraph." \
  --max-tokens 256 --temperature 0
```

The public MiniMax M3 BF16 checkpoint advertises MTP metadata but does not
publish `mtp` or `nextn` tensors, so use the released EAGLE-3 drafter for that
checkpoint.

MiniMax M3 also supports image/video prompts, MiniMax thinking tags, MiniMax
tool-call parsing, MSA index caches, and MXFP8 config loading. See
[`mlx_vlm/models/minimax_m3_vl/README.md`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/README.md)
for model-specific conversion and runtime notes.

## Python API

### Single sequence

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

### Batch generation

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

## Supported pairings

| Target | Drafter | Notes |
|--------|---------|-------|
| `Qwen/Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` | Text + image. ~2.5× speedup on code/reasoning. |
| `LiquidAI/LFM2.5-2.6B` | `LiquidAI/LFM2.5-2.6B-DSpark` | Text. Nine Markov-corrected proposals with exact LFM2 target verification. |
| `meta-models/Muse-Glimmer-30B` | `meta-models/Muse-Glimmer-30B-assistant` | Text + image. Native 5-layer, 16-token DFlash assistant. |
| `MiniMaxAI/MiniMax-M3` | `Inferact/MiniMax-M3-EAGLE3` | Text, image, and video target. Uses `--draft-kind eagle3`. |

The drafter is loaded via the shared `load_model` path. DFlash checkpoints are detected from `dflash_config` or the `muse_glimmer_assistant` model type; EAGLE-3 checkpoints are detected from `speculators_model_type` or EAGLE-3 architecture metadata. Native MTP sidecars for supported model families are detected from their `model_type`.

