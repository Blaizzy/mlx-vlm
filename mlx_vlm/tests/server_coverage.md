# Server test consolidation

Compared with `82d728b34434d919189c310db7d831d0b531299e`,
`test_server.py` decreases from **3,749 to 3,470 formatted lines**.
Including checks moved to the existing parser, speculative, and audio model
suites, the net reduction is **194 Python lines**. No production code changes.

- Share protocol tool-choice validation, stream decoding, message normalization,
  runtime setup, worker setup, audio backends, and WebSocket session setup.
- Combine generation arguments, timeout scenarios, settings errors, thinking
  markers, and tool-marker finalization into parameterized checks.
- Move MiniCPM syntax checks, speculative utility checks, and Nemotron audio
  buffering to their existing suites. Helpers remain inside test files.
- Preserve processor-free image templating and plain-object request handling:
  the coverage comparison identified these as distinct paths.
- Fold sampler forwarding, prompt metrics, tag counting, and settings replacement
  into related tests, retaining their assertions. Share streaming setup and
  function-result payloads; simplify CLI flag data and the nested STT fixture.

The second pass removes **69 lines** relative to `cbeb90fa`. Its 197 server cases
pass, and the full suite retains exactly the same 80,947 production lines and
12,834 branch outcomes as that commit. Four standalone tests are combined into
existing cases. Concurrency and shutdown-order checks remain intact.

## Validation

Both full-suite runs use the same isolated macOS arm64/Metal environment:
Python 3.12.14, MLX 0.32.2, Transformers 5.17.0, pytest 9.1.1, coverage 7.16.1.
Model downloads are disabled. Exact executed production line and branch sets
are compared; additions do not cancel losses.

| Metric | Before | After | Lost | Added |
| --- | ---: | ---: | ---: | ---: |
| Executed production lines | 80,943 | 80,947 | 0 | 4 |
| Executed branch outcomes | 12,831 | 12,834 | 0 | 3 |
| Passing tests | 1,791 | 1,790 | | |

Both runs have four skips and 39 passing subtests. The skips are two optional
VoiceChat checkpoint tests and two tests requiring PyTorch. Black, isort,
pyflakes, autoflake, and whitespace checks pass.

This retains all previously measured paths, not 100% coverage of production
code. Coverage includes imports and excludes tests, subprocess execution, native
kernels, and skipped scenarios; it does not establish assertion equivalence.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TOKENIZERS_PARALLELISM=false \
python -m coverage run --branch --source=mlx_vlm --omit='mlx_vlm/tests/*' \
  -m pytest -q mlx_vlm/tests --ignore=mlx_vlm/tests/test_smoke.py \
  -p pytest_timeout --timeout=90 --timeout-method=signal -ra
python -m coverage json -o server-suite-coverage.json
```
