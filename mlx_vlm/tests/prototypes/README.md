# Speculative 1K prototype

`speculative_1k.py` is an opt-in experiment, not a replacement for
`../test_speculative.py`. The normal suite still collects the original 519
speculative cases. The prototype imports production code and the shared tiny
configuration factories; it does not import, execute, or load the legacy tests.

## Size and checks

| Source | Readable lines, including imports and blank lines |
| --- | ---: |
| `speculative_1k.py` | 875 |
| `../speculative_fixtures.py` (counted in full) | 116 |
| Prototype and configuration fixtures | **991** |
| Common pytest setup, `../conftest.py` | 17 |
| Total including common pytest setup | **1,008** |

All case data lives in Python parameter tables; there is no additional JSON or
hidden test runner. The original suite plus configuration fixtures is 5,072 lines.
The prototype reduces that comparison by 80.5%, while dropping substantial checks.

The 314 collected cases use shared runners for:

- DFlash2, Muse Glimmer, and DSpark generation parity, seeded sampling, and request reset.
- Qwen3.5, GLM-5-Next, and DeepSeek V4 MTP generation and verifier/cache-commit parity.
- Quantized linear/argmax, fused projection, and MoE/hyperconnection numerical parity.
- MTP, DFlash, and Eagle3 round commit, generator close, and injected sampling failure.
- Acceptance budgets, sampler RNG isolation, and rotating drafter masks.
- Routing/compatibility, FP8 conversion, DeepSeek DSpark split/load/draft, Laguna
  checkpoint validation, and Eagle3 replay/draft-vocabulary mapping.

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
| Prototype alone | **314 passed** |
| Full suite with prototype replacing original speculative tests | **1,779 passed, 6 skipped, 51 subtests passed** |

Replacement collection was checked explicitly: 1,785 cases, including all 314
prototype cases and zero cases from `test_speculative.py`. Default collection
remains 1,990 cases and does not collect the prototype.

Retention means the intersection with the baseline's executed lines/branches;
newly exercised paths do not offset lost paths.

| Scope | Original statements retained | Original branches retained |
| --- | ---: | ---: |
| Speculative package, isolated suite comparison | 3,491 / 4,568 (**76.4%**) | 771 / 1,168 (**66.0%**) |
| All production reached by the isolated speculative suite | 11,541 / 16,698 (69.1%) | 1,751 / 2,672 (65.5%) |
| Coverage contributed by the original speculative suite beyond all other tests | 4,002 / 5,496 (**72.8%**) | 1,111 / 1,698 (**65.4%**) |
| Full production suite after substitution | 79,613 / 81,107 (98.2%) | 12,333 / 12,920 (95.5%) |

Replacing the original would lose **1,494 previously covered production statements
and 587 branch outcomes**, while adding 204 statements and 76 branch outcomes.
Whole-suite statement coverage would fall from 53.0555% to 52.2117%; branch
coverage would fall from 32.1409% to 30.8697%. The high whole-suite retention
includes the many unrelated tests that remain, so it overstates the similarity
of the two speculative suites.

Largest losses after substitution (these paths are not recovered by other tests):

| Production module under `mlx_vlm/` | Lost statements | Lost branch outcomes |
| --- | ---: | ---: |
| `speculative/mtp.py` | 156 | 70 |
| `speculative/eagle3.py` | 95 | 32 |
| `speculative/drafters/glm4_moe_lite_mtp/split.py` | 79 | 20 |
| `speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py` | 69 | 47 |
| `speculative/drafters/gemma4_dspark/gemma4_dspark.py` | 64 | 2 |
| `models/quantized_verifier.py` | 62 | 17 |
| `models/qwen3_5/language.py` | 62 | 38 |
| `speculative/drafters/gemma4_dspark/config.py` | 58 | 14 |
| `speculative/drafters/deepseek_v4_mtp/split.py` | 47 | 7 |
| `speculative/drafters/gemma4_assistant/masks.py` | 44 | 20 |

The prototype therefore demonstrates the 1K structure, but does not establish
similar coverage for the speculative subsystem. Adoption would require accepting
these losses or restoring targeted cases with a larger line budget. The existing
suite remains intact for that decision.

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
