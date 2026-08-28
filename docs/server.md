# Server (FastAPI)

Start the server:
```sh
mlx_vlm.server --port 8080

# Preload a model at startup (Hugging Face repo or local path)
mlx_vlm.server --model <hf_repo_or_local_path>

# Preload separate model kinds at startup
mlx_vlm.server --model <language_model> \
  --image-model <image_generation_model> \
  --tts-model <text_to_speech_model> \
  --stt-model <speech_to_text_model>

# Preload a model with adapter
mlx_vlm.server --model <hf_repo_or_local_path> --adapter-path <adapter_path>

# With trust remote code enabled (required for some models)
mlx_vlm.server --trust-remote-code

# Enable thinking mode by default for requests that do not override it
mlx_vlm.server --model Qwen/Qwen3.5-4B --enable-thinking

# Configure thinking defaults at startup
mlx_vlm.server --model Qwen/Qwen3.5-4B \
  --enable-thinking \
  --thinking-budget 512 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"

# Require bearer authentication for API endpoints
mlx_vlm.server --api-key <secret-token>
```

## Server Options

- `--model`: Preload a language model at server startup, accepts a Hugging Face repo ID or local path (optional, loads lazily on first request if omitted)
- `--image-model`: Preload an image generation model at server startup
- `--tts-model`: Preload a text-to-speech model at server startup
- `--stt-model`: Preload a speech-to-text model at server startup
- `--embedding-model`: Preload an embedding model at server startup
- `--reranker-model`: Preload a supported reranker model at server startup
- `--adapter-path`: Path for adapter weights to use with the preloaded model
- `--draft-model`: Speculative drafter path or HF id (e.g. `z-lab/Qwen3.8-27B-DFlash2`, `z-lab/Qwen3.5-4B-DFlash`, `RedHatAI/gemma-4-31B-it-speculator.eagle3`, `google/gemma-4-31B-it-assistant`, `Inferact/MiniMax-M3-EAGLE3`) — enables speculative decoding for ~2× or higher throughput
- `--draft-kind`: Drafter family — `dflash` (default), `eagle3`, or `mtp` (native/assistant MTP)
- `--draft-block-size`: Override the drafter's configured block size
- `--host`: Host address (default: `0.0.0.0`)
- `--port`: Port number (default: `8080`)
- `--trust-remote-code`: Trust remote code when loading models from Hugging Face Hub
- `--enable-thinking`: Enable thinking mode by default for requests that do not set `enable_thinking`
- `--thinking-budget`: Default maximum number of tokens allowed inside a thinking block
- `--thinking-start-token`: Default token that opens a thinking block
- `--thinking-end-token`: Default token that closes a thinking block (`--thinking-eos-token` is also accepted)
- `--kv-bits`: Number of bits for KV cache quantization (e.g. `8` for uniform, `3.5` for TurboQuant)
- `--kv-quant-scheme`: KV cache quantization backend (`uniform` or `turboquant`)
- `--kv-key-bits` / `--kv-value-bits`: Override the bit-width for keys or values individually (see [Per-tensor KV quantization](kv-cache-quantization.md#per-tensor-kv-quantization))
- `--kv-key-scheme` / `--kv-value-scheme`: Override the quantization backend for keys or values individually
- `--kv-group-size`: Group size for uniform KV cache quantization (default: `64`)
- `--max-kv-size`: Maximum KV cache size in tokens
- `--vision-cache-size`: Max number of cached vision features (default: `20`)
- `--log-progress-interval`: Decoded tokens between progress log messages; `0` disables periodic decode progress (default: `10`)
- `--api-key`: Bearer token required for inference, model discovery, and management endpoints
- `--log-level`: Logging level — `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)

At `INFO`, the server logs request start/completion, chunked-prefill progress,
time to first token, periodic decode throughput, and the final token counts. Set
`--log-level DEBUG` to emit decode progress for every token and add its token
number, token ID, and decoded text to the same log entry. Decode progress uses
`rate` for the instantaneous inter-token rate; decode completion uses the same
field name for aggregate decode throughput measured across completed token
intervals.

OpenAI-compatible streaming responses expose throughput under
`timings.predicted_per_second`. Token-bearing SSE chunks report the instantaneous
inter-token rate, while terminal and usage chunks report the aggregate rate as
`(tokens - 1) / (last_token_time - first_token_time)`. The first token reports
`null` because it has no preceding token interval.

You can also set trust remote code via environment variable:
```sh
MLX_TRUST_REMOTE_CODE=true mlx_vlm.server
```

The server provides multiple endpoints for different use cases and supports dynamic model loading/unloading with caching (one model at a time).

## Continuous Batching

The server supports continuous batching for higher throughput when handling multiple concurrent requests. New requests join the active batch immediately without waiting for existing requests to finish, and mixed batches of image and text-only requests are supported.

Continuous batching is enabled automatically when the server loads a model. You can pre-load a model at startup so it's ready to serve immediately:

```sh
mlx_vlm.server --port 8080 --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

Verify via the health endpoint:

```sh
curl http://localhost:8080/health
# {"status":"healthy","loaded_model":"...","apc_enabled":false}
```

If `--model` is omitted, the model is loaded on the first request.

## How It Works

- A dedicated generation thread runs a `BatchGenerator` that processes multiple requests in parallel
- Image requests are prefilled individually with their own vision embeddings, then join the shared decoding batch
- Text-only requests are batched together for efficient prefill
- After prefill, all requests decode together in a single batch, sharing GPU compute

## Available Endpoints

- `/models` and `/v1/models` - List models available locally
- `/chat/completions` and `/v1/chat/completions` - OpenAI-compatible chat-style interaction endpoint with support for images, audio, and text
- `/responses` and `/v1/responses` - OpenAI-compatible responses endpoint
- `/embeddings` and `/v1/embeddings` - OpenAI-compatible embeddings endpoint backed by native MLX embedding models — see [Embeddings & Reranking](embeddings-and-reranking.md)
- `/v1/rerank` - Rank text or multimodal documents by relevance to a query — see [Embeddings & Reranking](embeddings-and-reranking.md)
- `/audio/speech` and `/v1/audio/speech` - OpenAI-compatible text-to-speech endpoint backed by `mlx-audio` TTS models
- `/audio/transcriptions` and `/v1/audio/transcriptions` - OpenAI-compatible speech-to-text endpoint backed by `mlx-audio` STT models
- `/audio/translations` and `/v1/audio/translations` - OpenAI-compatible audio translation endpoint for STT models that expose a translation task
- `/v1/realtime` - WebSocket-based realtime full-duplex speech. See the [Nemotron VoiceChat guide](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/README.md#realtime-websocket-api) for supported models and usage.
- `/health` - Check server status
- `/metrics` and `/v1/metrics` - Inspect rolling request metrics, throughput, and runtime counters
- `/unload` - Unload all loaded model caches from memory

## Usage Examples

### List available models

```sh
curl "http://localhost:8080/models"
```

### Text Input

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you"
      }
    ],
    "stream": true,
    "max_tokens": 100
  }'
```

### Image Input

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
    "messages":
    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "This is today's chart for energy demand in California. Can you provide an analysis of the chart and comment on the implications for renewable energy in California?"
          },
          {
            "type": "input_image",
            "image_url": "/path/to/repo/examples/images/renewables_california.png"
          }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 1000
  }'
```

### Audio Input

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Describe what you hear in these audio files" },
          { "type": "input_audio", "input_audio": "/path/to/audio1.wav" },
          { "type": "input_audio", "input_audio": "https://example.com/audio2.mp3" }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 500
  }'
```

### Text-to-Speech

```sh
curl -X POST "http://localhost:8080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/pocket-tts",
    "input": "Hello from MLX VLM.",
    "voice": "fantine",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Speech-to-Text

```sh
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F model=mlx-community/parakeet-tdt-0.6b-v3 \
  -F file=@/path/to/audio.mp3 \
  -F response_format=json
```

### Multi-Modal (Image + Audio)

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_image", "image_url": "/path/to/image.jpg"},
          {"type": "input_audio", "input_audio": "/path/to/audio.wav"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

### Responses Endpoint

```sh
curl -X POST "http://localhost:8080/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is in this image?"},
          {"type": "input_image", "image_url": "/path/to/image.jpg"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

## Log Probabilities

The `/chat/completions` endpoint supports OpenAI-compatible per-token log probabilities. Pass `logprobs: true` (and optionally `top_logprobs: N`, up to 20) in the request:

```sh
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [{"role":"user","content":"Say hi in 3 words."}],
    "max_tokens": 8,
    "logprobs": true,
    "top_logprobs": 3
  }'
```

Each choice gets a `logprobs.content[]` list with one entry per generated token: `{token, logprob, bytes, top_logprobs: [{token, logprob, bytes}, ...]}`. Works for both streaming and non-streaming.

`top_logprobs` requires the server to be started with a non-zero cap on how many alternatives it will compute per token (default `0` = disabled, max `20`). Set it via the `--top-logprobs-k` flag or the `TOP_LOGPROBS_K` env var:

```sh
mlx_vlm.server --model mlx-community/Qwen2-VL-2B-Instruct-4bit --top-logprobs-k 5
# or
TOP_LOGPROBS_K=5 mlx_vlm.server --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

Per-request `top_logprobs` is clamped to `TOP_LOGPROBS_K`. When `TOP_LOGPROBS_K=0`, requests with `logprobs: true` still return chosen-token logprobs; only the `top_logprobs` list stays empty. Leaving the cap at `0` keeps the vocab-wide sort out of the decode graph, so deployments that don't need logprobs pay zero overhead.

## Structured Outputs

The `/v1/chat/completions` and `/v1/responses` endpoints support OpenAI-compatible `json_schema` structured outputs. The server constrains generation to the supplied JSON schema and supports both streaming and non-streaming responses.

You can define the schema with Pydantic:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnimalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    animal: Literal["dog", "cat", "bird", "unknown"]
    species: str = Field(max_length=60)
    description: str = Field(max_length=200)


schema = AnimalResult.model_json_schema()
```

Call the local server with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-4B-MLX-4bit",
    messages=[
        {"role": "user", "content": "Return a dog object."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "AnimalResult",
            "strict": True,
            "schema": schema,
        },
    },
)

result = AnimalResult.model_validate_json(response.choices[0].message.content)
print(result)
```

Example output:

```text
animal='dog' species='Canis lupus familiaris' description='A domesticated canine known for companionship and loyalty.'
```

Chat completions use top-level `response_format`. The same format works for text-only and multimodal requests:

```sh
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Identify the main animal in this image."},
        {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}
      ]
    }],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "AnimalResult",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "animal": {"type": "string", "enum": ["dog", "cat", "bird", "unknown"]},
            "species": {"type": "string", "maxLength": 60},
            "description": {"type": "string", "maxLength": 200}
          },
          "required": ["animal", "species", "description"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 256
  }'
```

Structured outputs are also supported with:

- Streaming chat completions by setting `"stream": true`
- The responses API via `text.format` on `/v1/responses`
- Text-only requests using the same `response_format` shape

Structured outputs are not currently supported with speculative decoding.

## Request Parameters

- `model`: Model identifier (required)
- `messages`: Chat messages for chat/OpenAI endpoints
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter
- `top_k`: Top-k sampling cutoff
- `min_p`: Min-p sampling threshold
- `repetition_penalty`: Penalty applied to repeated tokens
- `enable_thinking`: Override the server thinking-mode default for a request (`true` or `false`)
- `thinking_budget`: Maximum tokens allowed inside the thinking block
- `thinking_start_token`: Token that opens a thinking block
- `thinking_end_token`: Token that closes a thinking block
- `stream`: Enable streaming responses

## Live Settings (`/v1/settings`)

Read and change a curated set of server settings at runtime, without a
restart. `GET` lists the settings the server accepts; `PATCH` changes them.

```bash
# list the available settings and their current values
curl http://127.0.0.1:8080/v1/settings

# merge: only the settings you list are changed
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"kv_quant_scheme": "turboquant"}'

# replace: reset everything to its boot-time default, then apply these
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"op": "replace", "values": {"apc_enabled": true}}'
```

Changes take effect on the next request. Most settings reload the affected
model first — KV, APC, and speculative-decoding settings reload text models,
`vision_cache_size` reloads image models — while `max_kv_size` and
`token_queue_timeout` apply to new requests without a reload.

The response reports which settings were applied and which were rejected;
unknown names and invalid values are rejected and never applied.

## See also

- [Automatic Prefix Caching](prefix-caching.md) — reuse K/V across shared prefixes.
- [KV cache quantization](kv-cache-quantization.md) — `--kv-bits`, TurboQuant, per-tensor schemes.

