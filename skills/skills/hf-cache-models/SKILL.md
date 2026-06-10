---
name: hf-cache-models
description: Use this skill when the user wants to list, inspect, or report MLX-VLM-supported models available in the local Hugging Face cache directory, including models shown by the server /v1/models endpoint, cache-dir overrides, JSON output, or issue-ready cached model lists.
---

# HF Cache Models

Use this workflow to list locally cached Hugging Face models that MLX-VLM should treat as server-visible models.

## Supported Model Rule

Match the server `/v1/models` cache filter:

- repo type is `model`
- `main` revision exists in the cache
- `config.json` exists
- `tokenizer_config.json` exists
- either `model.safetensors.index.json` exists or at least one `*.safetensors` file exists

This is a cache/file-presence check. It does not load the model or prove generation works.

## Script

Use the bundled script instead of rewriting cache-scanning logic:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py
```

JSON output:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py --json
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

For server-visible verification, start the server and compare with:

```bash
curl http://127.0.0.1:8080/v1/models
```

If this becomes part of a bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
