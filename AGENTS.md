# AGENTS.md

Guidance for coding agents (and humans) working in the mlx-vlm repository. For the
narrative version, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Start here: skills

This repo ships an agent-skills bundle under [`skills/skills/`](skills/skills) that
encodes the project's own conventions for common workflows. **Load the relevant skill
before starting a task** — it tells you the right way to do it instead of guessing:

- [`add-new-model`](skills/skills/add-new-model) — port a new model to mlx-vlm.
- [`cli-inference`](skills/skills/cli-inference) — run inference from the CLI.
- [`server-inference`](skills/skills/server-inference) — the OpenAI/Anthropic-compatible server.
- [`convert-quantize`](skills/skills/convert-quantize) — convert and quantize checkpoints.
- [`benchmarking`](skills/skills/benchmarking) — measure performance.
- [`hf-cache-models`](skills/skills/hf-cache-models) — inspect the local Hugging Face cache.
- [`contributing`](skills/skills/contributing) — the development and contribution workflow.
- [`reproducible-github-issues`](skills/skills/reproducible-github-issues) — reproduce and file issues.

See the [Agent Skills](README.md#agent-skills) section of the README for installing them
into Claude Code, Codex, or Gemini.

## What this is

mlx-vlm is a local inference and fine-tuning engine for vision, text, audio, video,
omni, embedding, and re-ranking models on Apple silicon, built on [mlx](https://github.com/ml-explore/mlx).
It exposes a single CLI, an OpenAI/Anthropic-compatible server, and a Python API.

## Setup

Editable install from the repo root:

```bash
pip install -e .
```

## Repository layout

- `mlx_vlm/models/<model_type>/` — one directory per model. The main model file name
  must match the `model_type` in the model's Hugging Face `config.json`.
- `mlx_vlm/server/` — the OpenAI/Anthropic-compatible FastAPI server.
- `mlx_vlm/tests/` — tests; `model_cases.json` holds the per-model test configs.
- `docs/` — the documentation site (mkdocs).

## Adding a model

1. Start from an existing model in `mlx_vlm/models/` that is similar to the one you
   are porting.
2. Weights must be in `safetensors` format — convert if necessary.
3. Name the model file after the `model_type` in `config.json`.
4. If the model needs a `sanitize()` to convert weights from the PyTorch layout, it
   **must be idempotent**: `load_model` calls it on every checkpoint, including ones
   already in MLX layout, so running it on its own output has to be a no-op. Detect
   "already converted" explicitly per conversion (conv transpose, norm-weight shift,
   expert/qkv/gate_up splits, quant-scale rewrites, MTP/draft shard filtering) rather
   than relying on an earlier step having consumed its input. See CONTRIBUTING.md for
   the detection table.
5. Add a case to `mlx_vlm/tests/model_cases.json` and a test to `test_models.py`.

## Testing

Run the suite from the repo root:

```bash
python -m pytest -q mlx_vlm/tests
```

Prefer cheap synthetic / tiny-config tests. For `sanitize()` changes, verify a second
pass preserves weight keys, shapes, and values.

## Formatting

Formatting is enforced by `pre-commit` (`black`, `isort`, `autoflake` for Python;
`clang-format` for C++). Install once with `pre-commit install`, then let the hooks run
on commit, or run them manually:

```bash
pre-commit run --files file1.py file2.py   # specific files
pre-commit run --all-files                 # everything
```

Use `pre-commit` rather than calling `black`/`isort` directly, so versions match the
pinned config in `.pre-commit-config.yaml`.

## Pull requests

- Add tests for new code; every PR should have passing tests and at least one review.
- Keep changes scoped to the feature; gate new behavior so it is opt-in / disable-able
  and backward-compatible with existing checkpoints and configs.
