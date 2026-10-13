# MiniCPM 1B and 35B-A3B / DSpark SIMD prototype

This example compares ordinary token-by-token decoding with DSpark using the
same MiniCPM-V 4.7 target and SIMD arithmetic. It quantizes only the
DSpark-owned parameters and a separate draft vocabulary head. The target weights
are unchanged. Binding or resetting the drafter preserves its separate head.
The tested targets are BF16 1B and mixed-precision 35B-A3B: 4-bit projections
and experts, 8-bit routers, and BF16 embeddings and vocabulary head.

The verifier is opt-in. Ordinary model loading keeps the existing forward path.
For supported Qwen3.5 dense and MoE language models,
`language_model.use_simd_verifier = True` selects SIMD verification and singleton
decoding together. Keep this setting fixed throughout a request. The example additionally installs a
process-local adapter that pads the 1B model's 248,094-row vocabulary head by
two rows and removes those rows before sampling. This keeps an extra approximately
0.51 GB BF16 weight copy outside the checkpoint parameters.
The 35B-A3B head already has 248,144 aligned rows and reuses its existing weights.

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

## 1B measured results

M3 Ultra, BF16 target, 4-bit drafter and draft head, median of three runs with
1,024 output tokens. Both columns use the same SIMD target arithmetic.

| Fixture | Without DSpark | With DSpark | With DSpark, decode only |
|---|---:|---:|---:|
| Prose | 231.6 tok/s | 280.4 tok/s | 282.4 tok/s |
| Code | 231.8 tok/s | 399.3 tok/s | 403.0 tok/s |
| Image-conditioned | 223.9 tok/s | 265.5 tok/s | 277.4 tok/s |

The longer code completion emits 1,980 tokens at 402.5 tok/s
end-to-end (404.6 tok/s decode-only). These are fixture results,
not a general 400 tok/s guarantee. Prose/image use width four and code uses six.

Original 1B qualification (`c53bfece`): 678 verifier/speculative tests passed
on the qualified GPU. All 33 generation runs (39,528 tokens) match both serial decoding and the prior
prototype; all nine full-checkpoint logit/hidden/cache checks pass.
That M5 Max fallback run passed 116 tests and skipped 78 hardware-specific cases.

## 35B-A3B measured results

M3 Ultra, mixed-precision target described above, 4-bit drafter and draft head,
median of three runs with 1,024 output tokens. Both columns use the same SIMD
target arithmetic. The initial width sweep covered 3, 4, 6 and 8; fresh runs
with the final source confirmed width four for prose/image and six for code.

| Fixture | Without DSpark | With DSpark | With DSpark, decode only |
|---|---:|---:|---:|
| Prose | 79.4 tok/s | 113.6 tok/s | 114.8 tok/s |
| Code | 79.6 tok/s | 156.5 tok/s | 158.3 tok/s |
| Image-conditioned | 78.2 tok/s | 106.2 tok/s | 109.4 tok/s |

The selected policies improve full-request throughput by 1.43x, 1.97x and 1.36x,
respectively. The MoE checkpoint does not reach the 1B model's 400 tok/s result.
The result file includes prefill rates, individual repeats, complete-round
rates and output-token hashes. Exploratory measurements affected by a shared
GPU pause are excluded from these timing results.

The longer code run emits all 4,096 tokens at
79.2 tok/s without DSpark and 169.1 tok/s with DSpark
(169.8 tok/s decode-only), a 2.14x full-request gain.
All **33 final generation runs / 52,224 tokens** match serial decoding exactly.

Current validation: **863 verifier/speculative tests pass** on M3 Ultra.
The M5 Max fallback run passes **277 tests with 102 hardware-specific skips**.
All 11 full-checkpoint MoE cases match singleton logits, hidden states,
accepted KV/recurrent prefixes and the next logits after rollback, including
both sides of the 64K boundary. The original 1B checkpoint also passes its nine
regression cases. Hardware eligibility remains restricted to M3 Ultra.

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

This is a benchmark prototype, not a general CLI/server integration. The head
adapter is process-local. It is intended for one target model in the benchmark
process. Fixed widths were selected from measured fixtures, not a universal
policy. Routed experts keep the existing gather path; the routed MMA prototype
did not establish an end-to-end gain. Fused Markov, draft-fusion and
width-controller experiments without an end-to-end gain are also excluded.

See [results.json](results.json) for the original 1B results and
[moe_results.json](moe_results.json) for the MoE measurements and current validation.
