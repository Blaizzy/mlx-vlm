---
name: server-inference
description: Use this skill when the user wants to run or debug MLX-VLM server inference, including uv run mlx_vlm.server, /v1/models, /v1/chat/completions, /v1/responses, streaming, OpenAI-compatible clients, health checks, metrics, model unload/reload, adapters, trust-remote-code, and server request/response failures.
---

# Server Inference

Use this workflow for the FastAPI server and API-compatible inference.

## First Checks

1. Identify the server command, model, port, request endpoint, request body, and expected response.
2. Start with health/model-list checks before debugging generation.
3. Separate server startup failures from request-handling failures.
4. Keep streaming and non-streaming repros separate.

## Startup

```bash
uv run mlx_vlm.server \
  --model <model-or-path> \
  --port 8080
```

Useful startup flags include `--adapter-path`, `--trust-remote-code`, `--log-level`, `--enable-thinking`, `--thinking-budget`, `--draft-model`, `--draft-kind`, `--kv-bits`, `--kv-quant-scheme`, `--max-kv-size`, and `--vision-cache-size`.

## Minimal Checks

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/metrics
```

Minimal chat request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-or-path>",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 32,
    "stream": false
  }'
```

Minimal Responses API request:

```bash
curl -s http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-or-path>",
    "input": "Say hello.",
    "max_output_tokens": 32
  }'
```

## Debugging Rules

- Capture the server command, server logs, request body, response status, and response body.
- Test `/v1/models` after preload/load changes.
- For OpenAI client issues, reproduce with `curl` before blaming the client SDK.
- For streaming bugs, save raw event chunks and compare with non-streaming.
- For structured outputs, isolate the JSON schema and confirm whether the same request works without schema constraints.
- For tool calls, record the chat template/tool parser inferred by the loaded processor when possible.

## Validation

- For route/schema changes, run `uv run pytest mlx_vlm/tests/test_server.py -q`.
- For structured output changes, include `uv run pytest mlx_vlm/tests/test_structured.py -q`.
- For tool parser changes, include the relevant parser tests under `mlx_vlm/tests/test_*tool_parser.py`.
- If the result is a user-facing bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
