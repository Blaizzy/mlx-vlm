---
name: hf-cache-models
description: Use this skill when the user wants to list, inspect, or report cached, custom, or loaded MLX-VLM models through the server's model discovery endpoint.
---

# HF Cache Models

Query a running MLX-VLM server to list model candidates and their loaded status.
Use its configured URL and authentication; the examples assume the default local
address. See `Skill("mlx-vlm-skills:server-inference")` for server setup.

## List Models

```bash
curl http://127.0.0.1:8080/v1/models
```

The JSON response's `data` array contains model IDs and a `loaded` boolean.
Both `/models` and `/v1/models` include loaded models, the shared Hugging Face
cache, and configured custom directories by default.

## Supported Model Rule

The server checks discovered candidates with `mlx_vlm/server/model_discovery.py`:

- repo type is `model`
- readable model metadata exists in `config.json` or a pipeline `model_index.json`
- nonempty safetensors weights exist, including every shard referenced by any weight index
- all cached revisions are checked, preferring `main`; another revision is returned using its absolute snapshot path
- tokenizer metadata is not required

Discovery does not load weights, execute checkpoint Python, or prove generation
works. Architecture compatibility is checked when loading a model.

## Custom Model Directories

Each path can be a model folder or a parent containing model folders as immediate
children. Configure the server with repeated `--model-dir` options, or
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

- server URL and any custom directories requested
- number of model candidates
- exact model IDs
- which models have `loaded: true`

If this becomes part of a bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
