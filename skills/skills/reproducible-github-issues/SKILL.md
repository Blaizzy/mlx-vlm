---
name: reproducible-github-issues
description: Use this skill when the user wants to create, improve, or triage a reproducible GitHub issue for MLX-VLM, including bug reports from CLI inference, server inference, model loading, processors, media inputs, dependency setup, crashes, wrong outputs, or performance regressions.
---

# Reproducible GitHub Issues

Use this workflow to turn a failure into a concise, actionable MLX-VLM issue. Do not open a GitHub issue unless the user explicitly asks; otherwise produce issue-ready Markdown.

## Required Information

Collect or infer:

- MLX-VLM version or git commit.
- Install method: PyPI, editable checkout, branch, or wheel.
- Python version, OS version, machine/chip, and whether MLX Metal or MLX CUDA is in use.
- Exact model ID or local path.
- Whether the model is from Hugging Face cache, a local conversion, or a custom checkpoint.
- Exact `uv run` CLI command or server startup command.
- Exact request body for server issues.
- Input media facts: image dimensions, audio duration/sample rate, video duration/frame count, and whether the input can be shared.
- Expected behavior, actual behavior, and full error/traceback.

## Repro Minimization

1. Reduce to the smallest command or request that still fails.
2. Remove private paths, tokens, and unrelated environment variables.
3. Prefer `curl` over client SDKs for server repros.
4. Prefer one image/audio/video file before multi-input repros.
5. Use small public media or synthetic inputs when possible.
6. State whether the bug reproduces with a public model or only a private/local checkpoint.

## Issue Template

````markdown
### Summary

<One sentence describing the failure.>

### Environment

- MLX-VLM:
- Python:
- OS:
- Hardware:
- Install method:

### Model

- Model:
- Source: <HF cache | local path | converted checkpoint>
- Trust remote code: <yes/no>

### Reproduction

```bash
uv run mlx_vlm.generate <args>
```

For server issues:

```bash
uv run mlx_vlm.server <args>
```

```bash
<curl request>
```

### Expected Behavior

<What should have happened.>

### Actual Behavior

<What happened instead.>

### Logs / Traceback

```text
<trimmed traceback or relevant logs>
```

### Inputs

<Describe attached or shareable inputs. Include dimensions/duration when relevant.>
````

## Classification

- CLI inference: use `Skill("mlx-vlm-skills:cli-inference")` if a repro command is still missing.
- Server inference: use `Skill("mlx-vlm-skills:server-inference")` if a minimal request is still missing.
- Model-specific failures: include `config.json` `model_type`, processor class, and whether the family has a README in `mlx_vlm/models/`.

## Quality Bar

The issue is ready when a maintainer can run one command or one server command plus one request and see the same failure without asking for basic environment or model details.
