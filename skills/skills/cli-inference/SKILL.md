---
name: cli-inference
description: Use this skill when the user wants to run or debug MLX-VLM inference from the command line, including uv run mlx_vlm.generate, image/audio/video inputs, local model paths, Hugging Face model IDs, deterministic repro commands, and CLI errors around processors, prompts, model loading, or missing weights.
---

# CLI Inference

Use this workflow for `uv run mlx_vlm.generate` and related command-line inference tasks.

## First Checks

1. Identify the model ID or local path, modality, prompt, media files, and expected output.
2. Prefer an existing local model path or cached Hugging Face model when reproducing. Do not download a large model unless the user asks.
3. Check model-specific docs first when the family has a README under `mlx_vlm/models/<family>/README.md`.
4. Use `uv run mlx_vlm.generate --help` to verify current flags before giving a final command.

## Command Patterns

Text:

```bash
uv run mlx_vlm.generate \
  --model <model-or-path> \
  --prompt "Write a short answer." \
  --max-tokens 128
```

Image:

```bash
uv run mlx_vlm.generate \
  --model <model-or-path> \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 128
```

Audio or multimodal:

```bash
uv run mlx_vlm.generate \
  --model <model-or-path> \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "Describe what you see and hear." \
  --max-tokens 128
```

## Reproducibility Rules

- Include the exact command, model ID/path, media file type and size, Python version, package version or git commit, and full error.
- Use low-temperature or greedy settings when debugging quality or regressions.
- Bound output with `--max-tokens`.
- Preserve shell quoting exactly, especially prompts with JSON, XML-like thinking tokens, or newlines.
- If the failure depends on an image/audio/video file, record dimensions, duration, codec, and whether a small synthetic input reproduces it.

## Common Failure Routing

- `model_type` unsupported: inspect `config.json` and current folders under `mlx_vlm/models/`.
- Missing weights: check for `.safetensors` or `model.safetensors.index.json`.
- Processor/chat template errors: inspect `processing_*.py`, `prompt_utils.py`, and model-specific README examples.
- Media shape errors: test one image/audio/video item first, then multi-input cases.
- If the user needs an issue report rather than a fix, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.

## Validation

- Run the smallest command that exercises the requested modality.
- For code changes touching CLI parsing, run `uv run pytest mlx_vlm/tests/test_cli.py -q`.
- For generation behavior, prefer focused tests such as `test_generate.py`, `test_processors.py`, or a model-specific test file before broad suites.
