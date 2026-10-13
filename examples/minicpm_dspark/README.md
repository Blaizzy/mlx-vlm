# MiniCPM 1B / DSpark SIMD prototype

This example compares ordinary token-by-token decoding with DSpark using the
same BF16 MiniCPM-V 4.7 1B target and SIMD arithmetic. It quantizes only the
DSpark-owned parameters and a separate draft vocabulary head. The target weights
are unchanged. Binding or resetting the drafter preserves its separate head.

The verifier is opt-in. Ordinary model loading keeps the existing forward path.
For supported Qwen3.5 dense language models, `language_model.use_simd_verifier =
True` selects SIMD verification and singleton decoding together. Keep this
setting fixed throughout a request. The example additionally installs a
process-local adapter that pads MiniCPM's 248,094-row vocabulary head by two rows
and removes those rows before sampling. This keeps an extra approximately
0.51 GB BF16 weight copy outside the checkpoint parameters.

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

`serial` disables DSpark. Fixed widths include the anchor, so width six proposes
five tokens. `adaptive` runs the existing width controller. Omit the quantization
options to keep the drafter in BF16. `--mode native` selects the standard target
implementation; that implementation can produce a different token stream.

For longer code generation, use `--cases code --widths serial 6 --length 4096` and
a different output directory. EOS is honored: the tested fixture ends naturally
at 1,980 tokens. The budget must not be reported as the number of emitted tokens.

The harness warms each policy for 48 tokens and reverses policy order between
repeats. It writes configuration, complete token streams, per-round timings,
prefill/decode rates and median summaries. Full-request throughput includes
prefill, image processing and host work. No synchronization is added inside
normal decoding. End-of-request round accounting excludes un-emitted terminal
EOS and trailing proposals. The `serial` row is the token reference for each
fixture, and differing outputs fail the run.

## Measured results

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

Validation: 678 verifier/speculative tests pass on the qualified GPU. All 33
generation runs (39,528 tokens) match both serial decoding and the prior
prototype; all nine full-checkpoint logit/hidden/cache checks pass.
The M5 Max fallback run passes 116 tests and skips 78 hardware-specific cases.

## Correctness and limits

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python -m pytest \
  mlx_vlm/tests/test_simd_verifier.py mlx_vlm/tests/test_speculative.py -q

PYTHONPATH=. MLX_ENABLE_TF32=0 python examples/minicpm_dspark/verify.py \
  --model /path/to/MiniCPM-V-4.7-1B-mlx-bf16 \
  --output /tmp/minicpm-verifier-checks.json
```

The full-checkpoint script compares block and singleton logits, captured hidden
states, accepted KV/recurrent prefixes, and the next logits after rollback. It
covers empty, partial and full acceptance, plus attention dispatch boundaries.
The unit suite also covers dense BF16 and affine 4-bit projections, strided
inputs, fallback dispatch, fixed reduction order, and recurrent replay.

SIMD register ownership is checked at runtime on the qualified GPU. Reductions
use fixed eight-token tiles and dimension-dependent K partitions, FP32
accumulation and BF16 outputs. Equality with this path's singleton execution does
not imply bitwise equality with MLX's native matmul. Deferred recurrent replay
runs only after acceptance is known. Graph evaluation is pipelined every four
layers.

Attention grouping retains the existing MLX 0.32.3 / `applegpu_g15d` guard and
ratio-six qualification. The additional MiniCPM layout requires batch one,
eight query heads, two KV heads, head dimension 256, BF16, 2–8 query rows,
**fewer than 16,384 keys**, no sinks, and a causal string mask or no mask. It
splits causal groups at native dispatch boundaries. Longer MiniCPM contexts and
unsupported inputs use singleton attention; broader grouping failed numerical
qualification and is excluded.

This is a benchmark prototype, not a general CLI/server integration. The head
adapter is process-local. It is intended for one target model in the benchmark
process. Fixed widths were selected from measured fixtures, not a universal
policy. The 35B-A3B MoE model is excluded by the dense-model guard. The fused
Markov, draft-fusion and width-controller experiments that failed to show an
end-to-end gain are not included.

See [results.json](results.json) for measurements and validation from this branch.
