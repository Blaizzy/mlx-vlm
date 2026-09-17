# Speculative suite coverage

## Shared JSON config factory

Against `c4f513c3`, replace the Qwen, GLM and DeepSeek config wrappers and Python
factory lookup with `test_models.tiny_config`. Module/config-class metadata,
base settings and language/inference profiles live in `model_cases.json`.
Shared defaults are declared once. Each construction deep-copies the merged data;
explicit overrides take precedence. Training and checkpoint tests retain their
base settings, including DeepSeek's uncompressed cache configuration. Model,
drafter, checkpoint and transaction setup stays in `test_speculative.py`.

| File | Before | After | Change |
| --- | ---: | ---: | ---: |
| `test_speculative.py` | 1,652 | 1,635 | -17 |
| `test_models.py` | 664 | 662 | -2 |
| `test_trainer.py` | 717 | 716 | -1 |
| `model_cases.json` | 2,305 | 2,321 | +16 |
| **Combined** | **5,338** | **5,334** | **-4** |

This removes **20 Python lines** and adds **16 JSON lines**, saving **four combined
lines**. The main benefit is one reusable config factory, not a large LOC cut.
All 395 speculative cases remain, with identical test names, assertions and
parameterization. Captured base/language/inference/custom config values, model
parameter shapes and cache classes match the original factories; nested mutable
config values remain isolated between calls. Unrelated JSON settings are unchanged.

The speculative suite passes **395 cases**. Both full-suite runs pass **1,762
tests, four existing skips and 39 subtests**, with the same two dependency warnings.
Exact Python production execution sets are unchanged:

| Scope | Executed lines before/after | Branch outcomes before/after | Lost / added |
| --- | ---: | ---: | ---: |
| Speculative suite alone | 15,723 | 2,457 | 0 / 0 |
| Full default suite | 80,999 | 12,836 | 0 / 0 |

Full-suite statement coverage remains **52.9849%** and branch coverage **31.9319%**.
These measurements retain the current suite's coverage, including the earlier
intentional pruning documented below. They exclude test code, native kernels,
subprocesses and skipped optional checks. Validation uses the offline commands
below with a 90-second timeout, Python 3.12.14, MLX 0.32.2, Transformers 5.17.0,
pytest 9.1.1 and coverage 7.16.1. Black, isort, autoflake, pyflakes, Python 3.10
syntax parsing and whitespace checks pass.

## Speculative setup stays with its tests

Against `e345fc08`, speculative-only constructors, native checkpoint setup and
transaction doubles move from `test_models.py` back into `test_speculative.py`.
`dspark_source` and `glm_mtp_checkpoint_weights` are inline in their sole calling
tests, as is the two-override `dimensions` wrapper. General model construction
and the two DeepSeek/GLM config factories shared with training remain in
`test_models.py`. The complete shared profiles move out of the speculative JSON
section into `shared_configs`; their effective values are unchanged.

This restores the file boundary rather than maintaining the earlier 1,400-line
limit through relocation. No test, fixture or production module is added.

| File | Before | After | Change |
| --- | ---: | ---: | ---: |
| `test_speculative.py` | 1,399 | 1,652 | +253 |
| `test_models.py` | 929 | 664 | -265 |
| `model_cases.json` | 2,294 | 2,305 | +11 |
| **Combined** | **4,622** | **4,621** | **-1** |

All **395 speculative cases across 48 test functions** remain. An AST comparison
confirms unchanged assertions and parametrization after normalizing imported
helper references. The targeted speculative/model/training run passes **530
tests and eight subtests**. The full suite passes **1,762 tests, four skips and
39 subtests**, with the same two dependency deprecation warnings as the baseline.

Full-suite Python production coverage is **identical before and after: 80,999
executed lines and 12,836 branch outcomes**, with no lost or added paths. Both
runs use branch coverage over `mlx_vlm`, omit `mlx_vlm/tests/*`, exclude the manual
`test_smoke.py` runner, disable downloads and apply a 90-second per-test timeout.
These measurements exclude native kernels, subprocess execution and skipped
optional checks. Black, isort, autoflake, pyflakes and whitespace checks pass.
Earlier intentional test pruning remains documented below.

## Earlier refactor below 1,400 lines

The earlier refactor against `a1bbdca6` reduced `test_speculative.py` from
**2,027 to 1,399 formatted lines**. Shared model construction, synthetic checkpoint
tensors, and transaction doubles were placed in `test_models.py`; training
imported its two config factories from that module. Plain configuration and shape
data used `model_cases.json` under `speculative`. The ownership correction above
supersedes that layout. No production code changed.

This is mainly a separation of reusable setup from assertions, not a 628-line
reduction in the whole suite:

| File | Before | After | Change |
| --- | ---: | ---: | ---: |
| `test_speculative.py` | 2,027 | 1,399 | -628 |
| `test_models.py` | 650 | 929 | +279 |
| `model_cases.json` | 1,980 | 2,294 | +314 |
| **Combined** | **4,657** | **4,622** | **-35** |

The 413 previously collected speculative cases become **395 cases across 48
functions**. Three dispatch identity assertions now run inside the behavioral
round matrix. The 12 acceptance/budget combinations, three invalid-budget cases,
and uniform/empty-batch checks share one runner, retaining every scenario and
expected result. The unused Qwen MoE config factory and repeated successful
compatibility/block-size checks are removed. Checkpoint splitting and loading,
exact/tolerant numerical comparisons, and sampling call recorders are shared.

All **395 speculative cases pass**. The full default suite passes **1,773 tests**,
with **four existing skips and 39 passing subtests**, versus 1,791 tests before
consolidation. Both isolated and full-suite production execution sets are exactly
unchanged:

| Scope | Executed lines before/after | Branch outcomes before/after | Lost / added |
| --- | ---: | ---: | ---: |
| Speculative suite alone | 15,723 | 2,457 | 0 / 0 |
| Full default suite | 80,997 | 12,836 | 0 / 0 |

Black, isort, pyflakes, autoflake, JSON parsing, and whitespace checks pass.
Validation uses Python 3.12.14, MLX 0.32.2, Transformers 5.17.0, pytest 9.1.1,
and coverage 7.16.1, with downloads disabled. These are measured Python line and
branch sets; they exclude subprocess execution, native kernels, and skipped
optional checks. Numerical tolerances and exact sampled-token comparisons remain.
The earlier pruning described below is historical and is not reversed by this
refactor.

## Earlier compaction and intentional pruning

`test_speculative.py` is the default speculative suite. It replaces the previous
519-case module; the opt-in experiment and its separate report have been removed.
Production code is unchanged. These measurements compare the compact suite with
the previous suite at `5c3e00dc`, which remained unchanged through `bf85e718`.
The measurements below were recorded at `88324690`. At that stage, shared
configuration helpers lived directly in `test_speculative.py` and were imported
by training tests; the helper move preserved the test scenarios and assertions. The size breakdown
below describes the earlier layout, and full-suite counts predate later pruning.

## Size and validation

| Measure | Previous suite | Final suite |
| --- | ---: | ---: |
| Main Python module, including blank lines | 4,956 | 1,883 |
| Shared configuration helpers, then in a separate file | 116 | 116 |
| Combined readable lines | **5,072** | **1,999** |
| Including the common 17-line pytest setup | 5,089 | 2,016 |
| Speculative cases | 519 | **408 passed** |
| Full automated suite | 1,984 passed, 6 skipped | **1,873 passed, 6 skipped** |

The suite and fixtures shrink by **60.6%**. Case data stays in Python parameter
tables, with no hidden runner or additional JSON. All 361 cases from the compact
candidate remain represented; shared runners were extended and targeted checks
added. Default full-suite collection contains 1,879 cases, including the 408
speculative cases and no experimental cases. All 51 subtests pass. Formatting,
unused-import, and whitespace checks pass.

## Restored behavior

- Generation parity under explicit seeds and stateful sampling, request reset,
  proposal sampling independence, DSpark anchor positions, and stopping at rejection.
- Positioned and uniform batched acceptance, including mixed row positions and
  restoration of the original shared-KV state and draft-round counter.
- Full and sliding attention masks with different row offsets; rotating-window
  offsets, shared-KV padding, and DeepSeek mask widths.
- Cache commit and rollback across recurrent, rotating, native quantized, and
  TurboQuant caches; zero/full/ragged acceptance, stale transactions, failed
  verification, and cache filtering with aligned padding/positions.
- GLM draft-pool restoration and target-failure cleanup; Qwen quantized-cache
  rollback, LFM committed-prefix parity, MiniMax index caches, and Laguna rollback.
- Chunked-prefill feature preservation and an argmax fallback that must not append
  the same tokens twice.

These are shared behavioral contracts, not a one-to-one preservation of every
previous regression configuration. Qwen inference cases use 32-dimensional
recurrent heads. Float32 Qwen recurrent verification uses `atol=rtol=1e-6`, with
exact emitted tokens. Recurrent cache-prefix comparisons use `atol=1e-6, rtol=0`;
LFM ragged-decode parity retains the prior `atol=1e-4` check. Other verifier and
quantized-kernel comparisons require exact equality.

## Coverage retained

Retention is the intersection with previously executed statements or branch
outcomes. Newly exercised paths do not cancel lost paths.

| Scope | Statements retained | Branch outcomes retained |
| --- | ---: | ---: |
| Speculative package, each suite run alone | 4,278 / 4,568 (**93.7%**) | 1,040 / 1,168 (**89.0%**) |
| All production reached by each speculative suite alone | 15,457 / 16,698 (92.6%) | 2,338 / 2,672 (87.5%) |
| Coverage uniquely contributed beyond the other tests | 5,033 / 5,496 (**91.6%**) | 1,504 / 1,698 (**88.6%**) |
| Full production suite | 80,644 / 81,107 (99.4%) | 12,726 / 12,920 (98.5%) |

The rewrite loses **463 production statements and 194 branch outcomes** that the
other tests do not cover, while exercising 233 new statements and 95 new branch
outcomes. Whole-suite statement coverage changes from 53.0555% to **52.9050%**;
branch coverage changes from 32.1409% to **31.8946%**. High whole-suite retention
includes unrelated tests and does not imply equivalent speculative coverage.

The selected restorations recover 170 statements and 81 branches inside the
speculative package compared with the last compact candidate. Across production,
they recover 293 statements and 151 branches uniquely contributed by the old
suite. Remaining gaps include configuration/weight validation, specialized kernel
paths, drafter construction, server dispatch, and small compatibility wrappers.

Largest remaining losses:

| Production module under `mlx_vlm/` | Lost statements | Lost branch outcomes |
| --- | ---: | ---: |
| `speculative/drafters/laguna_dflash/dflash.py` | 43 | 0 |
| `models/fast_ops.py` | 40 | 12 |
| `models/glm5_next/language.py` | 31 | 6 |
| `speculative/drafters/qwen3_dflash/config.py` | 23 | 14 |
| `models/deepseek_v4/language.py` | 22 | 7 |
| `speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py` | 22 | 11 |
| `speculative/drafters/muse_glimmer_assistant/config.py` | 19 | 9 |
| `models/quantized_verifier.py` | 17 | 7 |
| `speculative/drafters/gemma4_dflash/__init__.py` | 16 | 0 |
| `models/mla.py` | 15 | 0 |

This is an intentional coverage tradeoff, not 100% preservation. The exact totals
above describe Python execution on the measured environment, not assertion
strength or Metal/C++ kernel coverage.

## Reproduce

Environment: macOS arm64/Metal, Python 3.12.14, MLX 0.32.2, Transformers 5.16.1,
pytest 9.1.1, coverage 7.16.1. Downloads were disabled. Two existing dependency
deprecation warnings remain; optional backend/checkpoint tests retain their skips.

```sh
python -m pytest -q mlx_vlm/tests/test_speculative.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TOKENIZERS_PARALLELISM=false COVERAGE_FILE=/tmp/speculative-full.coverage \
python -m coverage run --branch --source=mlx_vlm --omit='mlx_vlm/tests/*' \
  -m pytest -q mlx_vlm/tests --ignore=mlx_vlm/tests/test_smoke.py \
  -p pytest_timeout --timeout=90 --timeout-method=signal -ra
COVERAGE_FILE=/tmp/speculative-full.coverage \
python -m coverage json -o /tmp/speculative-full.json
```

For isolated coverage, run only `test_speculative.py` under the same coverage
command. For the other-tests baseline, add `--ignore=mlx_vlm/tests/test_speculative.py`
to the full run. Compare executed statement and branch sets by production filename
against the previous revision with identical dependencies, not rounded percentages.
