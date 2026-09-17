# APC compact prototype

`apc_compact.py` is a standalone alternative to `test_apc.py`: **1,499 formatted
Python lines versus 3,187**, a potential reduction of **1,688 lines (53.0%)**.
Its 35 test functions collect 97 cases; shared loops additionally exercise cache
families, memory-growth combinations, and repeated lookup sequences.

The original suite remains unchanged. The prototype has no `test_` filename and
is not collected by default. All new helpers are in this file. It uses the
existing `model_cases.json` APC profiles and the existing model-config and
DiffusionGemma helpers; no additional JSON or fixture module was introduced.

## Consolidation

- One manager factory owns memory budgets, disk namespaces, readers, writers,
  and cleanup. Specialized fixtures only select settings.
- Shared cache builders support construction, population, cloning, and batching.
  Recursive comparisons verify complete tensor state, dtypes, shapes, scalar
  metadata, and nested cache types across disk restoration.
- Source discovery produces **19 distinct cache layouts**, covering all layouts
  found by the preceding runtime-wrapper discovery. Model wrappers reuse local
  language-backbone factories. Model imports and discovery guards remain.
- A common memory-growth runner covers built-in, HyV4/ring, QSA, pooling, and
  MiniMax caches, preserving the previous seeds, chunk sizes, batch sizes,
  growth bounds, and MRoPE variants.
- Shared prompt construction retains LFM padded mixed prefill, Gemma4/Qwen3.5
  restored generation through streaming and batching, and DiffusionGemma suffix
  reuse. Numerical tolerances remain `1e-4` for LFM and `1e-5` for hybrid logits.
- Packed uniform/TurboQuant restoration shares a runner, including integer,
  fractional-budget, and split-codec cases. Guards prohibit dequantization and
  uniform requantization during snapshot, restoration, and merge.

Coverage-based pruning removes standalone subclass-only memory-profile checks,
the empty block-handle check, the source-text guard for the old KV-bits bypass,
and a duplicate multi-row snapshot/store case. Other checks are combined into
shared scenarios. Tenant hash isolation, subprocess hash stability, threaded disk
backpressure, memory admission before allocation, and long-prefix pressure remain.
The optional live checkpoint-backed join smoke stays in the original file and
is omitted from the prototype; its coverage was not measured.

## Measured coverage

Compared with `eda5edefe2853ee0185ba9916c7022641fee5df5`, using exact sets of
executed production statements and branch outcomes. Tests themselves are omitted.

| Run | Original | Prototype substituted | Lost | Added |
| --- | ---: | ---: | ---: | ---: |
| APC-only production statements | 39,647 | 39,655 | **0** | 8 |
| APC-only branch outcomes | 2,157 | 2,161 | **0** | 4 |
| Full-suite production statements | 80,919 | 80,926 | **0** | 7 |
| Full-suite branch outcomes | 12,819 | 12,822 | **0** | 3 |

This retains **100% of previously measured paths**, not 100% coverage of all
production code. Import-time execution is included. The comparison does not
measure Metal/C++ kernels, subprocess execution, or skipped checkpoint/backend
scenarios, and does not establish equivalence of every original assertion.

- Original APC: **182 passed, 1 skipped**.
- Prototype alone: **97 passed**.
- Full suite with the prototype substituted: **1,791 passed, 4 skipped,
  39 passing subtests**, with two existing dependency deprecation warnings.
- JUnit output confirms exactly **97 prototype cases and zero original APC
  cases** in the replacement run. The four skips are two VoiceChat checkpoint
  tests and two optional PyTorch reference tests.
- Black, isort, pyflakes, autoflake, and whitespace checks pass.
- Environment: macOS arm64/Metal, Python 3.12.14, MLX 0.32.2,
  Transformers 5.16.1, pytest 9.1.1, coverage 7.16.1; downloads disabled.

## Reproduction

Run the prototype explicitly:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TOKENIZERS_PARALLELISM=false \
python -m coverage run --branch --source=mlx_vlm --omit='mlx_vlm/tests/*' \
  -m pytest -q mlx_vlm/tests/prototypes/apc_compact.py \
  -p pytest_timeout --timeout=90 --timeout-method=signal -ra
python -m coverage json -o apc-prototype-coverage.json
```

For the full replacement comparison, pass explicit file paths. Passing the parent
test directory together with the nonstandard prototype filename can cause pytest
to omit the prototype during directory collection.

```python
import os
import subprocess
import sys
from pathlib import Path

env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TOKENIZERS_PARALLELISM="false")
files = sorted(str(p) for p in Path("mlx_vlm/tests").glob("test_*.py")
               if p.name not in {"test_apc.py", "test_smoke.py"})
files.append("mlx_vlm/tests/prototypes/apc_compact.py")
subprocess.run([
    sys.executable, "-m", "coverage", "run", "--branch", "--source=mlx_vlm",
    "--omit=mlx_vlm/tests/*", "-m", "pytest", "-q", *files,
    "-p", "pytest_timeout", "--timeout=90", "--timeout-method=signal", "-ra",
    "--junitxml=apc-prototype-full.xml",
], env=env, check=True)
subprocess.run([sys.executable, "-m", "coverage", "json",
                "-o", "apc-prototype-full.json"], env=env, check=True)
```

For the baseline, replace the appended prototype path with `test_apc.py`. Compare
the exact `(file, line)` and `(file, branch-start, branch-end)` sets in the JSON
reports: additions do not cancel losses.
