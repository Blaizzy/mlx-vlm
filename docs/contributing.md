# Contributing

Contributions are welcome — new models, bug fixes, documentation, and performance
work. This page covers the development workflow. For the conventions a coding agent
should follow on a specific task, see [AGENTS.md](https://github.com/Blaizzy/mlx-vlm/blob/main/AGENTS.md)
and the [skills bundle](https://github.com/Blaizzy/mlx-vlm/tree/main/skills/skills)
(porting a model, running the CLI/server, converting and quantizing, benchmarking, and
reproducing issues).

## Development install

Clone the repo and install it in editable mode:

```bash
git clone https://github.com/Blaizzy/mlx-vlm
cd mlx-vlm
pip install -e .
```

Install the formatting hooks once so they run on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Repository layout

- `mlx_vlm/models/<model_type>/` — one directory per model. The main model file name
  matches the `model_type` in the model's Hugging Face `config.json`.
- `mlx_vlm/server/` — the OpenAI/Anthropic-compatible server.
- `mlx_vlm/tests/` — the test suite. `model_cases.json` holds the per-model test configs.
- `docs/` — this documentation site.

## Adding a model

1. Start from the existing model in `mlx_vlm/models/` that is closest to the one you are
   porting — most ports are a variation of an existing architecture.
2. Make sure the weights are in [`safetensors`](https://huggingface.co/docs/safetensors/index)
   format; convert if necessary.
3. Name the model file after the `model_type` in `config.json`.
4. Add a case to [`mlx_vlm/tests/model_cases.json`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/tests/model_cases.json)
   so the model is exercised by the shared checks in `test_models.py`.

Keep new behavior opt-in and backward-compatible, so existing checkpoints and configs
keep loading unchanged.

### `sanitize()` must be idempotent

If a model needs a `sanitize()` to convert weights from the PyTorch layout, note that
`load_model` calls it on **every** checkpoint — including ones already in MLX layout.
`sanitize()` must therefore be a no-op when run on its own output. Detect "already
converted" explicitly for each transform rather than relying on an earlier step having
consumed its input key:

| conversion | detect it by |
|---|---|
| conv transpose | target layout (`check_array_shape`, or compare against `self.<conv>.weight.shape`) |
| norm-weight shift (`v + 1.0`) | a marker only an unconverted checkpoint carries; `v.ndim == 1` is **not** one, since it holds before and after |
| expert / qkv / gate_up splits and merges | absence of the source key |
| quantization-scale rewrites | the scale dtype or grouping already being the MLX one |
| MTP or draft shard filtering | presence of the shard keys |

Getting this wrong fails two ways: a shape-changing conversion applied twice dies loudly
at weight-load, while a value-changing one (a norm shift, a scale rewrite) loads fine and
generates garbage. For sanitizer changes, verify a second pass preserves the weight keys,
shapes, and values.

## Running tests

From the repository root:

```bash
python -m pytest -q mlx_vlm/tests
```

Prefer cheap synthetic / tiny-config tests, and check exactness against the original path
in the degenerate limit (for example, a new feature equals the flat baseline when it is
disabled).

## Formatting

Formatting is enforced by `pre-commit` — `black`, `isort`, and `autoflake` for Python, and
`clang-format` for C++. If you did not `pre-commit install`, run it manually before pushing:

```bash
pre-commit run --files file1.py file2.py   # specific files
pre-commit run --all-files                 # everything
```

## Pull requests

- Fork the repo and open a pull request against `main`.
- If you add code that should be tested, add tests.
- Every PR should have passing tests and at least one review.
- Run `pre-commit` before pushing so CI formatting checks pass.

We use GitHub issues to track bugs — please include enough detail to reproduce. See
[Report Issues](report_issues.md) for what to include.
