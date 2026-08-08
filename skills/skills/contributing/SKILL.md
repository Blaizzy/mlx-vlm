---
name: contributing
description: Use this skill when the user wants to contribute to MLX-VLM — opening a PR, where model code/config/tests go, backward-compatible config args, running the test suite, code formatting and the pre-commit hooks (black, clang-format), and PR expectations (tests, review, perf evidence). Use it to set up a change so it passes review.
---

# Contributing to MLX-VLM

Use this workflow to shape a contribution so it lands cleanly. It reflects the current `mlx_vlm/` layout (note: `CONTRIBUTING.md` still points at an older `src/` layout — the real paths are below).

## Setup

```bash
git clone https://github.com/<you>/mlx-vlm && cd mlx-vlm
uv pip install -e .            # editable install (or: pip install -e .)
```

## Where things go

- **Model code:** `mlx_vlm/models/<model_type>/`, and the file name matches the `config.json` `model_type` (e.g. `llava`). A new kernel or helper goes in **its own file in that model directory**, not bolted into `language.py`. To add a whole new model, use `Skill("mlx-vlm-skills:add-new-model")`.
- **Config args:** add to the model's `ModelConfig` (in `<model>/config.py`) with an inline `# comment`, and keep them **backward-compatible** — a default of `None`/`0`/`False` must reproduce the original behavior so existing checkpoints and configs are unaffected. Gate new features so they are opt-in.
- **Tests:** add to `mlx_vlm/tests/test_models.py` as a **class per feature** (e.g. `TestMyFeature`). Do not create a standalone `test_<feature>.py` — repo convention keeps model tests in `test_models.py`. Prefer cheap synthetic/random-weight tests (tiny configs) and check exactness against the original path in the degenerate limit (feature disabled == baseline).

## Run the tests

```bash
cd mlx_vlm && uv run --with pytest python -m pytest tests/test_models.py -q -k "<ClassName>"
```

## Formatting — install and run pre-commit

**Before opening a PR, install the pre-commit hooks once and run them.** The repo uses `pre-commit` (black for Python, clang-format for C++); PRs are expected to be hook-clean.

```bash
pip install pre-commit
pre-commit install                 # install the git hooks (run once per clone)

# run on the files you changed
pre-commit run --files <file1.py> <file2.py>
# or check everything
pre-commit run --all-files
```

You can also format a single file directly:

```bash
black <file>.py
# clang-format -i <file>.cpp   # for C++
```

Reminder: run `pre-commit run --all-files` (or at least on your changed files) as the **last step before pushing** — an unformatted diff will fail CI/review.

## PR expectations

1. Fork and open the PR against `main`.
2. New code that should be tested comes with tests (see above).
3. Every PR needs passing tests and at least one review.
4. Include **perf evidence** for anything performance-sensitive — a self-contained, reproducible benchmark. Use `Skill("mlx-vlm-skills:benchmarking")` for the fork-vs-main table format.
5. Keep the change scoped and opt-in; don't regress existing checkpoints.

## Routing

- Adding a model → `Skill("mlx-vlm-skills:add-new-model")`.
- Converting/quantizing weights → `Skill("mlx-vlm-skills:convert-quantize")`.
- Perf numbers for the PR → `Skill("mlx-vlm-skills:benchmarking")`.
- Filing a bug found while contributing → `Skill("mlx-vlm-skills:reproducible-github-issues")`.
