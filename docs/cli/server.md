# mlx_vlm.server — OpenAI-compatible server

A FastAPI server with continuous batching that exposes OpenAI-compatible chat, responses, embeddings, reranking, audio, image, and realtime endpoints. Models can be pre-loaded at startup or discovered and loaded on demand.

## Synopsis

```
mlx_vlm.server [OPTIONS]
```

The console script `mlx_vlm.server` also works.

## Examples

Start the server on a port and pre-load a vision-language model:

```
mlx_vlm.server \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --port 8080
```

Require a bearer token on inference, discovery, and management endpoints:

```
mlx_vlm.server \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --api-key sk-local-secret
```

Turn on thinking mode by default and cap the thinking block:

```
mlx_vlm.server \
  --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --enable-thinking \
  --thinking-budget 1024
```

Serve with a speculative drafter:

```
mlx_vlm.server \
  --model mlx-community/Qwen3.5-VL-9B-Instruct-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --draft-block-size 8
```

## Options

### Model preloading

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `None` | Pre-load a language model at startup (e.g. `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`). |
| `--image-model` | `None` | Pre-load an image generation model at startup. |
| `--tts-model` | `None` | Pre-load a text-to-speech model at startup. |
| `--stt-model` | `None` | Pre-load a speech-to-text model at startup. |
| `--embedding-model` | `None` | Pre-load an embedding model at startup. |
| `--reranker-model` | `None` | Pre-load a supported reranker model at startup. |
| `--model-dir` | `None` | Additional model directory, or parent directory of model folders; repeat for multiple paths (overrides `MLX_VLM_MODEL_PATHS`). |
| `--adapter-path` | `None` | Adapter weights to load with the model. |
| `--trust-remote-code` | `False` | Trust remote code when loading models from the Hugging Face Hub. |

### Server & runtime

| Flag | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Host for the HTTP server. |
| `--port` | `8080` | Port for the HTTP server. |
| `--api-key` | `None` | Optional bearer token required for inference, model discovery, and management endpoints (maps to `MLX_VLM_SERVER_API_KEY`). |
| `--reload` | `False` | Enable auto-reload for development. |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `--log-progress-interval` | `10` | Decoded tokens between progress log messages; `0` disables periodic decode progress. |
| `--max-num-seqs` | `None` (unbounded) | Maximum sequences decoded concurrently in the continuous batch; extra requests wait in the queue (maps to `MLX_VLM_MAX_NUM_SEQS`). |
| `--vision-cache-size` | `20` | Max number of cached vision features. |
| `--prefill-step-size` | `2048` | Tokens per prefill step. |

### Generation defaults

| Flag | Default | Description |
| --- | --- | --- |
| `--max-tokens` | `2048` | Maximum number of tokens to generate. |
| `--top-logprobs-k` | `None` (`0` = disabled) | Server-side cap for per-token `top_logprobs` (0-20); maps to the `TOP_LOGPROBS_K` env var. |

### Thinking

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-thinking` | `False` | Enable thinking mode by default for requests that do not set `enable_thinking` explicitly. |
| `--thinking-budget` | `None` | Default maximum tokens allowed inside a thinking block (requests can override with `thinking_budget`). |
| `--thinking-start-token` | `None` | Default token that opens a thinking block (requests can override with `thinking_start_token`). |
| `--thinking-end-token` (alias `--thinking-eos-token`) | `None` | Default token that closes a thinking block (requests can override with `thinking_end_token`). |

### Speculative decoding

| Flag | Default | Description |
| --- | --- | --- |
| `--draft-model` | `None` | Speculative drafter path or HF id (e.g. `z-lab/Qwen3.5-4B-DFlash`, `google/gemma-4-31B-it-assistant`). |
| `--draft-kind` | `None` (auto-detected) | Drafter family: `dflash`, `eagle3`, or `mtp` (Gemma 4); defaults to the drafter's HF `model_type`. |
| `--draft-block-size` | `None` | Override the drafter's configured block size. |

### KV cache & quantization

| Flag | Default | Description |
| --- | --- | --- |
| `--kv-bits` | `None` | Number of bits for KV cache quantization (e.g. `3.5` for TurboQuant). |
| `--kv-key-bits` | `None` | Override the TurboQuant key bit-width (defaults to `floor(--kv-bits)`). |
| `--kv-value-bits` | `None` | Override the TurboQuant value bit-width (defaults to `ceil(--kv-bits)`). |
| `--kv-key-scheme` | `None` | Override the KV quantization backend for keys only (`uniform`, `turboquant`). |
| `--kv-value-scheme` | `None` | Override the KV quantization backend for values only (`uniform`, `turboquant`). |
| `--kv-quant-scheme` | `uniform` | KV cache quantization backend (`uniform`, `turboquant`). |
| `--kv-group-size` | `64` | Group size for uniform KV cache quantization. |
| `--max-kv-size` | `None` | Maximum KV cache size in tokens. |
| `--quantized-kv-start` | `5000` | Start index for quantized KV cache. |

### Memory

| Flag | Default | Description |
| --- | --- | --- |
| `--expert-cache-gb` | `None` (70% of the GPU's recommended working set) | For an `mlx_vlm.moe_offload` checkpoint, bound the resident routed-expert set to this many GB; ignored for a normal, non-offloaded checkpoint. |

## See also

- [Server guide](../server.md) — full endpoint reference and curl examples
- [Prefix caching](../performance/prefix-caching.md)
- [KV cache quantization](../performance/kv-cache-quantization.md)
