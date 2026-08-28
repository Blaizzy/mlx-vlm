---
name: hf-cache-models
description: Use this skill when the user wants to list, inspect, or report MLX-VLM model candidates available in the local Hugging Face cache directory, including the server's opt-in hf-cache discovery mode, cache-dir overrides, JSON output, or issue-ready cached model lists.
---

# HF Cache Models

Use this workflow to list locally cached Hugging Face models that MLX-VLM can expose when Hugging Face cache discovery is explicitly enabled.

## Supported Model Rule

Match the server `/v1/models` filter used with `--model-discovery hf-cache`:

- repo type is `model`
- `main` revision exists in the cache
- `config.json` exists
- `tokenizer_config.json` exists
- either `model.safetensors.index.json` exists or at least one `*.safetensors` file exists

This is a cache/file-presence check that mirrors the server's opt-in `hf-cache` discovery mode (`mlx_vlm/server/app.py`). It does not load the model, prove generation works, or affect the default `served` listing. Pass `--check-arch` to additionally require that mlx-vlm ships an architecture for the `model_type` — this narrows the list from a *cache candidate* to *probably loadable* (folder-name match; it does not resolve `MODEL_REMAPPING` aliases, so use it as a strong hint, not proof).

## Script

Use the bundled script instead of rewriting cache-scanning logic:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py
```

JSON output:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py --json
```

Only models mlx-vlm can actually load (architecture present, not just files present):

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py --check-arch
```

Custom cache directory:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py \
  --cache-dir /path/to/huggingface/cache
```

## Reporting

When reporting the result, include:

- cache directory used, if non-default
- number of supported models
- exact model IDs
- whether the list came from the script or from `curl http://127.0.0.1:8080/v1/models`

For Hugging Face cache-discovery verification, start the server with
`--model-discovery hf-cache` and compare with:

```bash
curl http://127.0.0.1:8080/v1/models
```

If this becomes part of a bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
