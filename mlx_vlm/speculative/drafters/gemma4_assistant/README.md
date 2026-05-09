# Gemma 4 Assistant Drafter (MTP)

MLX port of Google's Gemma 4 **Multi-Token Prediction (MTP)** drafter for
speculative decoding. Reference:
[ai.google.dev/gemma/docs/mtp](https://ai.google.dev/gemma/docs/mtp/mtp).

## What it is

A small, 4-layer "assistant" model trained to draft several candidate tokens
per round; the full Gemma 4 target verifies them in a single forward pass.
Accepted tokens advance, rejected ones (and everything after) are discarded.
Quality matches the target at temperature 0 (byte-identical greedy output).

The drafter is tightly coupled to the target's internals:

- **KV-cache sharing** — every drafter layer is `is_kv_shared_layer=True` and
  reads K/V from the target's last full-attention and last sliding-attention
  layers. The drafter has **no KV cache of its own**; its only recurrent
  state is the target's last hidden, projected through `post_projection`.
- **Cross-attention from constant position** — the drafter's queries are
  RoPE-rotated at the bonus token's absolute position and held constant
  across all draft steps within a block.
- **Hidden+token concatenation** — drafter input each step is
  `concat([target_embed(last_token), last_hidden_state], dim=-1)` of shape
  `[B, 1, 2 * backbone_hidden_size]`, projected to drafter hidden size by
  `pre_projection`.

## Supported pairings

| Target                                | Drafter                                              | LM head                |
| ------------------------------------- | ---------------------------------------------------- | ---------------------- |
| `mlx-community/gemma-4-E2B-it-bf16`               | `mlx-community/gemma-4-E2B-it-assistant-bf16`                  | centroid (sparse)      |
| `mlx-community/gemma-4-E4B-it-bf16`               | `mlx-community/gemma-4-E4B-it-assistant-bf16`                  | centroid (sparse)      |
| `mlx-community/gemma-4-26B-A4B-it-bf16`           | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`              | tied dense             |
| `mlx-community/gemma-4-31B-it-bf16`               | `mlx-community/gemma-4-31B-it-assistant-bf16  `                  | tied dense             |

For E2B / E4B drafters, `use_ordered_embeddings=True` and the LM head is a
**centroid-routed sparse softmax** (`MaskedEmbedder`): the drafter scores
2048 token clusters, materialises the top-K (default 32) clusters' tokens
(~4096 of 262144), and scatters those logits back into a full-vocab tensor —
non-selected positions filled with `min(selected) - 1` so they lose any
argmax / sampling competition.

## Files

- `config.py` — `Gemma4AssistantConfig` (HF-compatible, flattened).
- `gemma4_assistant.py` — `Gemma4AssistantDraftModel` (forward, `bind`,
  `set_shared_kv`, `draft_block`, `sanitize`).
- `masked_embedder.py` — centroid-routed sparse LM head for E2B / E4B.
- `masks.py` — bidirectional full / SWA masks for the drafter forward.
- `parity_check.py` — fake-target smoke test.

## Usage

The drafter is auto-discovered by HF `model_type == "gemma4_assistant"`;
just pass `--draft-model` and `--draft-kind mtp` to `mlx_vlm.generate`:

```bash
uv run python -m mlx_vlm.generate \
    --model mlx-community/gemma-4-31B-it-bf16 \
    --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
    --draft-kind mtp \
    --draft-block-size 4 \
    --prompt "Explain speculative decoding in 3 sentences." \
    --max-tokens 256 --temp 0
```

`--draft-block-size` is the number of speculatively drafted tokens per
round (google calls this `num_assistant_tokens`). The first token of the
block is the most recently accepted bonus, so the drafter actually
generates `block_size - 1` candidates each round.

Programmatic use:

```python
from mlx_vlm.utils import load
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.generate import generate_step

model, processor = load("mlx-community/gemma-4-31B-it-bf16")
drafter = load_drafter("mlx-community/gemma-4-31B-it-assistant-bf16", kind="mtp")

for tok, _ in generate_step(
    input_ids, model, None, None,
    max_tokens=256,
    draft_model=drafter,
    draft_kind="mtp",
    draft_block_size=4,
):
    ...
```

## Performance

Measured on Apple Silicon (M3 Max, 96GB RAM), 17-token prompt, max 64–96 tokens, greedy
(`temp=0`), output byte-identical to the no-drafter baseline.

Best `block_size` per (target, batch):

| Target  | B  | best bs | tot tok/s | speedup vs no-drafter |
|---------|----|---------|-----------|-----------------------|
| 26B-A4B | 4  | 3       | 85.5      | **3.94×**             |
| 26B-A4B | 8  | 3       | 165.1     | **1.55×**             |
| 31B     | 4  | 3       | 17.1      | **2.29×**             |
| 31B     | 8  | 2       | 21.4      | **1.41×**             |
| E4B     | 4  | 4       | 62.1      | **1.56×**             |
| E4B     | 8  | 2       | 115.9     | 1.07×                 |
| E4B     | 16 | —       | —         | drafter slower (≤1.0×)|

The drafter is most attractive on large/slow targets (26B-A4B, 31B) where
target forward time dominates. On the small E4B target, target forward is
already cheap and at high batch sizes the drafter's per-step overhead
exceeds the speedup it buys.

## Caveats

- **Sampling.** Greedy (temp 0) is verified byte-identical. Stochastic
  sampling works but acceptance rates drop because drafter and target
  draws diverge.
- **Multimodal prompts.** Image / audio prefill runs through the target
  unchanged; speculative decoding only kicks in on the text-decode tail,
  so multimodal works but the drafter only ever sees text tokens.
- **Sliding-window masks.** The bidirectional SWA mask in `masks.py`
  short-circuits to `None` when `kv_len <= sliding_window`, which is the
  only regime `RotatingKVCache` ever produces. Long-prompt mask paths are
  effectively dead code today.
- **Batched generation.** Continuous-batching support is in
  `_mtp_rounds_batch` (`mlx_vlm/generate.py`). For targets whose KV caches
  don't implement `.filter()`, finished rows are kept in the batch and
  simply stop emitting; throughput doesn't shrink with retired rows.
