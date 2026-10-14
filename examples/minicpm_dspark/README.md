# MiniCPM 1B and 35B-A3B / DSpark SIMD verification

This example compares ordinary token-by-token decoding with DSpark using the
same MiniCPM-V 4.7 target and SIMD arithmetic. It quantizes only the
DSpark-owned parameters and a separate draft vocabulary head. The target weights
are unchanged. Binding or resetting the drafter preserves its separate head.
The tested targets are BF16 1B and mixed-precision 35B-A3B: 4-bit projections
and experts, 8-bit routers, and BF16 embeddings and vocabulary head.

The verifier is opt-in. Ordinary model loading keeps the existing forward path.
For supported Qwen3.5 dense and MoE language models,
`language_model.use_simd_verifier = True` selects SIMD verification and singleton
decoding together. Keep this setting fixed while requests or prompt caches are
in use. Hardware, dtype and shape checks retain the ordinary fallback.

The verifier prepares unaligned BF16 projections internally. The 1B vocabulary
head needs two extra rows; its approximately 0.51 GB padded copy is cached on
that projection, outside checkpoint parameters, and rebuilt when the weight
array is replaced. Padded logits are removed before sampling. The already
aligned 35B-A3B head reuses its weights. The shared verifier holds no model
weights. KV/recurrent rollback continues through the existing cache transaction
and deferred-replay APIs.

DSpark owns its optional quantized proposal head. Binding and resetting reuse
that head for unchanged target weights and quantization settings, and rebuild
it when the target weights or bias change. Rebinding also refreshes shared
embeddings while preserving embeddings owned by the drafter checkpoint.
Preparing either derived head leaves the sampling RNG untouched.

## Existing loader, CLI and server paths

Normal loaders can enable SIMD from the target checkpoint's `text_config`:

```json
{"use_simd_verifier": true}
```

The DSpark checkpoint can configure its separate proposal head through these
fields in `dflash_config` (or at the top level):

```json
{"proposal_head_bits": 4, "proposal_head_group_size": 64}
```

`proposal_head_bits` defaults to `null`; supported values are 4 and 8. This
option requires an untied, unquantized `nn.Linear` target head. It creates a
separate quantized copy without changing the target. Draft backbone/Markov
weights use the checkpoint's usual quantization configuration.

The existing `--model` / `--draft-model` CLI and server options consume those
checkpoints through the shared loaders. No head adapter or binding patch is
needed. Python callers can set `language_model.use_simd_verifier` and
`draft.config.proposal_head_bits` before starting generation. Legacy MiniCPM
drafters with a flat `projector_type: "dspark"` now use the same loader path.
To reproduce fixed widths, set `block_size_policy: "fixed"` and
`runtime_block_size` in the drafter configuration; width four suits the measured
prose/image fixtures and six suits code.

## Reproduce

Use the repository's dependencies with MLX and mlx-metal **0.32.3** on the
qualified Apple M3 Ultra (`applegpu_g15d`). Run from the repository root with a
converted BF16 target and a local BF16 DSpark checkpoint:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python examples/minicpm_dspark/benchmark.py \
  --model /path/to/MiniCPM-V-4.7-1B-mlx-bf16 \
  --draft /path/to/MiniCPM-V-4.7-1B-DSpark \
  --draft-bits 4 --draft-head-bits 4 \
  --widths serial 4 6 --cases text code image \
  --length 1024 --runs 3 --output-dir /tmp/minicpm-dspark-1024
```

For the MoE checkpoint and its matching drafter:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python examples/minicpm_dspark/benchmark.py \
  --model /path/to/MiniCPM-V-4.7-35B-A3B-mlx-4bit \
  --draft /path/to/MiniCPM-V-4.7-35B-A3B-DSpark \
  --draft-bits 4 --draft-head-bits 4 \
  --widths serial 4 6 --cases text code image \
  --length 1024 --runs 3 --output-dir /tmp/minicpm-moe-1024
```

`serial` disables DSpark. Fixed widths include the anchor, so width six proposes
five tokens. `adaptive` runs the existing width controller. Omit the quantization
options to keep the drafter in BF16. `--mode native` selects the standard target
implementation; that implementation can produce a different token stream.

For longer code generation, use `--cases code --widths serial 6 --length 4096` and
a different output directory. EOS is honored: the 1B fixture ends naturally
at 1,980 tokens. The budget must not be reported as the number of emitted tokens.

The harness warms each policy for 48 tokens and reverses policy order between
repeats. It writes configuration, complete token streams, per-round timings,
prefill/decode rates and median summaries. Full-request throughput includes
prefill, image processing and host work. No synchronization is added inside
normal decoding. End-of-request round accounting excludes un-emitted terminal
EOS and trailing proposals. The `serial` row is the token reference for each
fixture, and differing outputs fail the run.

## Measured results

M3 Ultra, MLX/Metal 0.32.3, TF32 off, batch one, greedy, median of three
runs with 1,024 emitted tokens. Both columns use the same SIMD target
arithmetic. Drafters and separate proposal heads use affine 4-bit weights,
group size 64. The target precision is described above.

| Model | Fixture | Without DSpark | With DSpark | With DSpark, decode only |
|---|---|---:|---:|---:|
| 1B | Prose | 227.1 tok/s | 276.8 tok/s | 278.8 tok/s |
| 1B | Code | 227.6 tok/s | 395.7 tok/s | 399.3 tok/s |
| 1B | Image-conditioned | 219.4 tok/s | 261.0 tok/s | 272.7 tok/s |
| 35B-A3B | Prose | 79.5 tok/s | 113.8 tok/s | 115.0 tok/s |
| 35B-A3B | Code | 79.7 tok/s | 156.5 tok/s | 158.4 tok/s |
| 35B-A3B | Image-conditioned | 78.1 tok/s | 106.1 tok/s | 109.3 tok/s |

Full-request rates include prefill, image processing and host work. Prose/image
use width four and code uses six. The initial MoE width sweep covered 3/4/6/8;
these fresh runs cover serial/4/6 for both models. These fixture results do not
establish a general 400 tok/s guarantee. Throughput remains close to the
pre-refactor measurements; this change moves ownership into the shared library.

The longer code measurements in [results.json](results.json) and
[moe_results.json](moe_results.json) predate the ownership refactor and have not
been repeated here. With a 4,096-token budget, the 1B fixture stopped naturally
at 1,980 tokens and reached 402.5 tok/s end-to-end. The MoE fixture emitted all
4,096 tokens at 79.2 tok/s without DSpark and 169.1 tok/s with DSpark
(169.8 tok/s decode-only).

## Current validation

- M3 Ultra: **887 passed** across the verifier and speculative suites.
- M5 Max fallback: **784 passed, 103 hardware-specific skips** in the same suites.
- CLI/server speculative, cancellation and cache tests: **12 passed**.
- MiniCPM model contract tests: **15 passed**.
- All **54 benchmark runs / 55,296 emitted tokens** match serial decoding and
  the pre-refactor implementation exactly. Target parameters remain unchanged.
- All nine 1B and eleven MoE full-checkpoint checks match singleton logits,
  hidden states, accepted KV/recurrent prefixes and next logits after rollback,
  with zero maximum logit error. MoE checks include both sides of the 64K boundary.
- Actual CLI and server entrypoints load the configured 1B target and DSpark
  checkpoint through shared loaders. The CLI succeeds, and two consecutive
  server requests produce identical messages. This smoke test uses a BF16 draft
  backbone and 4-bit proposal head.

The lifecycle tests cover proposal-head reuse across reset, invalidation after
weight/bias changes and disabling quantization, target rebinding, shared versus owned embeddings,
and preservation of the sampling RNG. Verifier tests cover projection-owned
padding, removal of padded vocabulary logits before masked argmax, and fallback.
The existing cache transaction and deferred-replay tests remain the rollback
contract; this change adds no second cache mechanism.

## Correctness and limits

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python -m pytest \
  mlx_vlm/tests/test_simd_verifier.py mlx_vlm/tests/test_speculative.py -q

PYTHONPATH=. MLX_ENABLE_TF32=0 python examples/minicpm_dspark/verify.py \
  --model /path/to/MiniCPM-V-4.7-1B-mlx-bf16 \
  --output /tmp/minicpm-verifier-checks.json

PYTHONPATH=. MLX_ENABLE_TF32=0 python examples/minicpm_dspark/verify.py \
  --model /path/to/MiniCPM-V-4.7-35B-A3B-mlx-4bit \
  --capture-layers 3 11 19 27 35 --long-context \
  --output /tmp/minicpm-moe-verifier-checks.json
```

The full-checkpoint script compares block and singleton logits, captured hidden
states, accepted KV/recurrent prefixes, and the next logits after rollback. It
covers empty, partial and full acceptance, plus attention dispatch boundaries.
The unit suite also covers dense BF16 and affine 4-bit projections, strided
inputs, fallback dispatch, fixed reduction order, and recurrent replay.
MoE cases cover eight selected experts, BF16/4-bit expert weights, 8-bit routers,
and optional router-score normalization. The full-checkpoint script accepts
`--capture-layers 3 11 19 27 35` for the 35B drafter and `--long-context` to
check rollback at the 64K attention boundary.

SIMD register ownership is checked at runtime on the qualified GPU. Reductions
use fixed eight-token tiles and dimension-dependent K partitions, FP32
accumulation and BF16 outputs. Equality with this path's singleton execution does
not imply bitwise equality with MLX's native matmul. Deferred recurrent replay
runs only after acceptance is known. Graph evaluation is pipelined every four
layers.

Attention grouping retains the existing MLX 0.32.3 / `applegpu_g15d` guard and
ratio-six qualification. The additional MiniCPM layout requires batch one,
eight or sixteen query heads, two KV heads, head dimension 256, BF16,
2–8 query rows, **fewer than 16,384 keys**, no sinks, and a causal string mask
or no mask. It splits causal groups at native dispatch boundaries. Longer
MiniCPM contexts and unsupported inputs use singleton attention; broader grouping failed numerical
qualification and is excluded.
The 35B layout groups at most four queries: grouping five changes native
attention reductions even below the context boundary.

Verifier preparation and proposal-head ownership are library behavior shared
by the Python, CLI and server paths. The example only configures those paths
and instruments benchmark rounds. Fixed widths were selected from measured
fixtures, not a universal policy. Routed experts keep the existing gather path; the routed MMA prototype
did not establish an end-to-end gain. Fused Markov, draft-fusion and
width-controller experiments without an end-to-end gain are also excluded.

See [integration_results.json](integration_results.json) for current measurements,
individual repeats, prefill/decode/round rates, token hashes and validation.
[results.json](results.json) and [moe_results.json](moe_results.json) preserve the
earlier 1B and MoE measurements.
