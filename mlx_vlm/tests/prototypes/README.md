# Speculative prototype: 1K plus 500 lines

`speculative_1k.py` is an opt-in experiment, not a replacement for
`../test_speculative.py`. The normal suite still collects the original 519
speculative cases. The prototype imports production code and the shared tiny
configuration factories; it does not import, execute, or load the legacy tests.

## Size and checks

| Source | Readable lines, including imports and blank lines |
| --- | ---: |
| `speculative_1k.py` | 1,372 |
| `../speculative_fixtures.py` (counted in full) | 116 |
| Prototype and configuration fixtures | **1,488** |
| Common pytest setup, `../conftest.py` | 17 |
| Total including common pytest setup | **1,505** |

All case data lives in Python parameter tables; there is no additional JSON or
hidden test runner. The original suite plus configuration fixtures is 5,072 lines.
The prototype reduces that comparison by 70.7%, while dropping some checks.
This revision adds **497 lines** to the initial 991-line prototype, within the
additional 500-line budget. All original prototype checks remain unchanged.

The 361 collected cases use shared runners for:

- DFlash2, Muse Glimmer, and DSpark generation parity, seeded sampling, and request reset.
- Qwen3.5, GLM-5-Next, and DeepSeek V4 MTP generation and verifier/cache-commit parity.
- Quantized linear/argmax, fused projection, and MoE/hyperconnection numerical parity.
- MTP, DFlash, and Eagle3 round commit, generator close, and injected sampling failure.
- Acceptance budgets, sampler RNG isolation, and rotating drafter masks.
- Routing/compatibility, FP8 conversion, DeepSeek DSpark split/load/draft, Laguna
  checkpoint validation, and Eagle3 replay/draft-vocabulary mapping.
- Padded Qwen prefill, batched MTP commit/filter, rotating and quantized cache rollback.
- Gemma DSpark configuration/layers, positioned deferred sampling, Eagle EOS verification,
  and adaptive block sizing.
- Native GLM/DeepSeek checkpoint layouts, GLM weight fusion, wide quantized verification,
  shared-KV padding, and sparse logits.

The generic Qwen verifier cases use 32-dimensional recurrent heads so the inference
Metal kernel can execute. Float32 Qwen recurrent outputs use `atol=rtol=1e-6`;
selected tokens must still match exactly. Other verifier cases and the quantized
kernel comparisons require exact equality. These are shared contract scenarios,
not a one-to-one preservation of every original regression or configuration.

## Measured coverage

Baseline: commit `5c3e00dc`, unchanged production code. Measurements ran on macOS
arm64/Metal with Python 3.12.14, MLX 0.32.2, pytest 9.1.1, and coverage 7.16.1.
Hugging Face downloads were disabled. Coverage excludes `mlx_vlm/tests/*` and
measures Python statements and branch outcomes, not Metal/C++ execution.

| Run | Result |
| --- | --- |
| Original speculative suite alone | 519 passed |
| Prototype alone | **361 passed** |
| Full suite with prototype replacing original speculative tests | **1,826 passed, 6 skipped, 51 subtests passed** |

Replacement collection was checked explicitly: 1,832 cases, including all 361
prototype cases and zero cases from `test_speculative.py`. Default collection
remains 1,990 cases and does not collect the prototype.

Retention means the intersection with the baseline's executed lines/branches;
newly exercised paths do not offset lost paths.

| Scope | Original statements retained | Original branches retained |
| --- | ---: | ---: |
| Speculative package, isolated suite comparison | 4,108 / 4,568 (**89.9%**) | 959 / 1,168 (**82.1%**) |
| All production reached by the isolated speculative suite | 12,510 / 16,698 (74.9%) | 2,057 / 2,672 (77.0%) |
| Coverage contributed by the original speculative suite beyond all other tests | 4,740 / 5,496 (**86.2%**) | 1,353 / 1,698 (**79.7%**) |
| Full production suite after substitution | 80,350 / 81,107 (99.1%) | 12,574 / 12,920 (97.3%) |

The extra 497 lines recover **617 previously covered statements and 188 branch
outcomes inside the speculative package**. That raises retention from 76.4% to
89.9% for statements and from 66.0% to 82.1% for branches.

For coverage uniquely contributed by the original suite across production, the
remaining gap is **756 statements and 345 branch outcomes**, down from 1,494 and
587. This recovers 738 statements and 242 branches that other tests do not cover.

The raw whole-suite comparison loses 757 statements and 346 branch outcomes,
while adding 230 statements and 92 branch outcomes. One additional statement and
branch difference occurs in the unrelated realtime server close/cancel path,
which was exercised by the other-tests-only run. Whole-suite statement coverage
would fall from 53.0555% to 52.7108%; branch coverage from 32.1409% to 31.5090%.
The high whole-suite retention includes unrelated tests and overstates the
similarity of the two speculative suites.

Largest losses after substitution (these paths are not recovered by other tests):

| Production module under `mlx_vlm/` | Lost statements | Lost branch outcomes |
| --- | ---: | ---: |
| `speculative/mtp.py` | 99 | 47 |
| `speculative/drafters/laguna_dflash/dflash.py` | 43 | 0 |
| `models/fast_ops.py` | 40 | 12 |
| `models/deepseek_v4/language.py` | 39 | 17 |
| `models/cache.py` | 32 | 17 |
| `models/glm5_next/language.py` | 32 | 7 |
| `models/minimax_m3_vl/language.py` | 30 | 12 |
| `speculative/common.py` | 27 | 11 |
| `models/laguna/language.py` | 24 | 17 |
| `speculative/drafters/qwen3_dflash/config.py` | 23 | 14 |

The expanded prototype improves coverage within the additional 500-line budget,
but does not restore 100% of the previous coverage. The existing suite remains
the default; adopting this prototype would accept the measured gaps above.

## Run it

Run only the prototype:

```sh
python -m pytest -q mlx_vlm/tests/prototypes/speculative_1k.py
```

Measure a real replacement run, collecting the opt-in filename explicitly through
pytest's filename pattern and excluding the original file:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TOKENIZERS_PARALLELISM=false COVERAGE_FILE=/tmp/speculative-1k.coverage \
python -m coverage run --branch --source=mlx_vlm --omit='mlx_vlm/tests/*' \
  -m pytest -q mlx_vlm/tests \
  --ignore=mlx_vlm/tests/test_smoke.py \
  --ignore=mlx_vlm/tests/test_speculative.py \
  -o 'python_files=test_*.py speculative_1k.py' \
  -p pytest_timeout --timeout=90 --timeout-method=signal -ra
COVERAGE_FILE=/tmp/speculative-1k.coverage \
python -m coverage json -o /tmp/speculative-1k-coverage.json
```

For the baseline, omit the speculative ignore and filename-pattern override. For
isolated measurements, pass only the corresponding Python file to pytest. To
measure the original suite's unique contribution, run the other tests with the
speculative ignore and the default filename pattern. Compare sets of executed
statements and branches by production filename, not just rounded percentages.
