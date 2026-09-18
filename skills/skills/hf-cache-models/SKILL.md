---
name: hf-cache-models
description: Use this skill when the user wants to list, inspect, or report MLX-VLM model candidates available in the local Hugging Face cache directory, including server model discovery, cache-dir overrides, JSON output, or issue-ready cached model lists.
---

# HF Cache Models

Use this workflow to list locally cached Hugging Face models that MLX-VLM exposes through model discovery.

## Supported Model Rule

Match the server `/v1/models` cache filter:

- repo type is `model`
- readable model metadata exists in `config.json` or a pipeline `model_index.json`
- nonempty safetensors weights exist, including every shard referenced by any weight index
- all cached revisions are checked, preferring `main`; another revision is returned using its absolute snapshot path
- tokenizer metadata is not required

The script and server share the metadata and weight checks in `mlx_vlm/server/model_discovery.py`. They do not load weights, execute checkpoint Python, or prove generation works. The server also includes loaded models and marks each entry with a `loaded` boolean. Pass `--check-arch` to additionally require that mlx-vlm ships an architecture for the `model_type` — this narrows the list from a *cache candidate* to *probably loadable* (folder-name match; it does not resolve `MODEL_REMAPPING` aliases, so use it as a strong hint, not proof).

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

Custom model folders outside the Hugging Face cache:

```bash
uv run python skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py \
  --model-dir /Volumes/Models --model-dir ~/my-custom-model
```

Each path can be a model folder or a parent containing model folders as immediate
children. The server accepts the same repeated `--model-dir` option, or
`MLX_VLM_MODEL_PATHS` with paths separated by `os.pathsep` (`:` on macOS/Linux).

For directories scoped to one API request, use repeated `model_dir` query
parameters on `/models` or `/v1/models`. Paths are on the server's filesystem and
are added to its configured directories without persisting them:

```bash
curl --get http://127.0.0.1:8080/v1/models \
  --data-urlencode 'model_dir=/Volumes/Models' \
  --data-urlencode 'model_dir=/Users/me/my custom model'
```

## Reporting

When reporting the result, include:

- cache directory used, if non-default
- number of supported models
- exact model IDs
- whether the list came from the script or from `curl http://127.0.0.1:8080/v1/models`

For Hugging Face cache-discovery verification, start the server and compare with:

```bash
curl http://127.0.0.1:8080/v1/models
```

If this becomes part of a bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
