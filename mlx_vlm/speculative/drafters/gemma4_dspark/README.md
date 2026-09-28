# Gemma 4 DSpark drafter

`gemma4_dspark` runs DeepSeek's DSpark self-speculative decoding against Gemma 4
targets. Like `deepseek_v4_dspark`, it is a **backbone variant of the
model-agnostic `dspark` drafter**: it reuses the shared DSpark machinery —
`VanillaMarkov` block sampling, the confidence head, and the `dflash` round loop
with its target-hidden tap — and swaps only the Qwen-style draft layers for
Gemma 4 blocks.

Unlike `deepseek_v4_dspark`, the drafter is published as a standalone repository
rather than inside the base checkpoint's `mtp.*` namespace, so there is no
`split.py`.

- Drafter: `deepseek-ai/dspark_gemma4_12b_block7`
- Target: any `gemma4` / `gemma4_unified` checkpoint with 48 text layers

## Reuse over reimplementation

The draft layers compose `models.gemma4.language` rather than restating Gemma 4:
`MLP` supplies the GeGLU feed-forward, and the projection geometry follows
`gemma4.language.Attention`. Only two pieces are genuinely drafter-specific.

`Gemma4DSparkAttention` subclasses `DFlashAttention` to keep the context /
proposal split, overriding the projections for Gemma 4 geometry:
`global_head_dim` (512) on full-attention layers instead of `head_dim`, a unit
softmax scale, `num_global_key_value_heads` (1) for MQA, and `attention_k_eq_v`,
under which keys and values share one projection so the checkpoint ships no
`v_proj`. That last case is what the `_project_kv` hook on `DFlashAttention`
exists for.

`Gemma4DSparkDecoderLayer` applies Gemma 4's sandwich norms — input,
post-attention, pre-feedforward, post-feedforward — plus the per-layer
`layer_scalar`, where the generic DFlash layer has only two norms.

## Checkpoint contract

Two details differ from the Qwen-style DSpark checkpoints:

The config declares the generic `model_type: gemma4_text` and identifies itself
through the `Gemma4DSparkModel` architecture tag, so `get_model_and_args` routes
on the tag. It also publishes the DSpark fields flat rather than under
`dflash_config`, which `Gemma4DsparkConfig.from_dict` reads either way.

RoPE follows Gemma 4's per-layer-type parameters — `partial_rotary_factor` 0.25
at theta 1e6 for full attention — and rotates `global_head_dim`. The generic
DFlash helper would instead use `head_dim` and a flat `rope_theta`.

The drafter also carries untied `embed_tokens` and `lm_head`, so `bind()` keeps
its own rather than adopting the target's as DFlash does.

## Drafter precision

The drafter is published in bf16 (6.9 GB) and is worth quantizing: the target
block pass dominates a round, but the drafter still reads its whole 262144-row
`lm_head` every round, and it keeps its own untied head rather than the
target's. At 8-bit it moves 3.6 GB instead of 6.9 GB per round, and the
proposals do not change -- mean acceptance and the generated tokens are
identical to bf16 at every context measured.

`mlx_vlm.convert` cannot be used here: the drafter repo ships no tokenizer or
processor, so processor loading fails. Quantize the weights directly instead:

```python
import json, pathlib
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.utils import load_model, get_model_path

src = get_model_path("deepseek-ai/dspark_gemma4_12b_block7")
out = pathlib.Path("./dspark_gemma4_12b_block7-8bit")
out.mkdir(parents=True, exist_ok=True)

drafter = load_model(src)
nn.quantize(drafter, group_size=64, bits=8)
mx.eval(drafter.parameters())

mx.save_safetensors(
    str(out / "model.safetensors"), dict(tree_flatten(drafter.parameters()))
)
config = json.load(open(src / "config.json"))
config["quantization"] = {"group_size": 64, "bits": 8}
json.dump(config, open(out / "config.json", "w"), indent=2)
```

4-bit is not faster than 8-bit and costs acceptance (5.40 to 5.10 at 2k), so
8-bit is the recommended setting.

## Exactness

Speculation does not reproduce base decoding token-for-token on every prompt.
Measured over six chat prompts at 256 tokens, greedy:

| prompt | base vs base | vs dspark | vs dspark, exact verify |
| --- | --- | --- | --- |
| reasoning | 0/256 | 208/256 | 208/256 |
| code | 0/256 | 141/256 | 0/256 |
| factual | 0/256 | 0/256 | 0/256 |
| summarize | 0/256 | 159/256 | 159/256 |
| creative | 0/256 | 0/256 | 108/256 |
| longform | 0/256 | 49/256 | 57/256 |

The cause is the target's own block forward, not the drafter. The divergence is
unchanged across block sizes 2, 4 and 8; unchanged when acceptance is forced to
zero so that every emitted token is one the target produced itself; and
identical for bf16 and 8-bit drafters down to the first-divergence index. What
is left is that block verification evaluates the target over `L >= 2` positions
where base decoding evaluates it over one, and the model sits near enough to an
argmax tie on these prompts that the last-bit difference flips tokens.

`models/gemma4/speculative_verifier.py` narrows this but does not close it -- it
removes the divergence on one prompt and introduces it on another -- because it
restores singleton numerics for the linear projections and attention but not for
every op in the block. It stays **opt-in** via
`text_config.exact_speculative_verify`: its per-position attention loop costs
roughly half of decode throughput (30.5 vs 77.4 tok/s at 2k), and on this
architecture most projections cannot reach the singleton-equivalent Metal
kernels at all, since Gemma 4's hidden size of 3840 fails their `K % 512 == 0`
precondition and q/k/v/gate/up fall back to a per-token loop.

## Known limitation

Speculative gain is 1.7-2.4x except for prompts near `sliding_window` (1024),
where it drops to ~0.5x. Sliding layers hold a `RotatingKVCache`, and
`update_and_fetch` routes any multi-token write to `_update_concat`, which
rebuilds the whole cache to keep it in the temporal order the positional causal
mask requires. That is O(window) per verification round instead of O(block),
and it bites hardest once the ring saturates but total model work is still
small. The cost is shared infrastructure: it applies to every block-speculative
path — `dflash`, `eagle3`, `mtp` — on any sliding-window target, not just this
drafter. Fixing it needs ring-aware masking rather than a change here.
