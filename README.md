[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (VLMs with audio and video support) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Thinking Budget](#thinking-budget)
  - [Speculative Decoding](#speculative-decoding)
    - [DFlash (Qwen3.5)](#dflash-qwen35)
    - [Gemma 4 MTP](#gemma-4-mtp)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
  - [Server (FastAPI)](#server-fastapi)
    - [Continuous Batching](#continuous-batching)
    - [Automatic Prefix Caching (APC)](#automatic-prefix-caching-apc)
    - [KV Cache Quantization](#kv-cache-quantization)
- [Activation Quantization (CUDA)](#activation-quantization-cuda)
- [Multi-Image Chat Support](#multi-image-chat-support)
  - [Supported Models](#supported-models)
  - [Usage Examples](#usage-examples)
- [Model-Specific Documentation](#model-specific-documentation)
- [Vision Feature Caching](#vision-feature-caching)
- [TurboQuant KV Cache](#turboquant-kv-cache)
- [Distributed Inference](#distributed-inference)
- [Fine-tuning](#fine-tuning)

## Model-Specific Documentation

Some models have detailed documentation with prompt formats, examples, and best practices:

| Model | Documentation |
|-------|---------------|
| DeepSeek-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/README.md) |
| DeepSeek-OCR-2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md) |
| DOTS-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| DOTS-MOCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| GLM-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm_ocr/README.md) |
| Phi-4 Reasoning Vision | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4_siglip/README.md) |
| MiniCPM-o | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/README.md) |
| Phi-4 Multimodal | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4mm/README.md) |
| MolmoPoint | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/README.md) |
| Moondream3 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/moondream3/README.md) |
| Gemma 4 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md) |
| Falcon-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/falcon_ocr/README.md) |
| Granite Vision 3.2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite_vision/README.md) |
| Granite 4.0 Vision | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite4_vision/README.md) |

## Installation

The easiest way to get started is to install the `mlx-vlm` package using pip:

```sh
pip install -U mlx-vlm
```

## Usage

### Command Line Interface (CLI)

Generate output from a model using the CLI:

```sh
# Text generation
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Hello, how are you?"

# Image generation
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg

# Audio generation (New)
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you hear" --audio /path/to/audio.wav

# Multi-modal generation (Image + Audio)
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you see and hear" --image /path/to/image.jpg --audio /path/to/audio.wav
```

#### Thinking Budget

For thinking models (e.g., Qwen3.5), you can limit the number of tokens spent in the thinking block:

```sh
mlx_vlm.generate --model mlx-community/Qwen3.5-2B-4bit \
  --thinking-budget 50 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>" \
  --enable-thinking \
  --prompt "Solve 2+2"
```

| Flag | Description |
|------|-------------|
| `--enable-thinking` | Activate thinking mode in the chat template |
| `--thinking-budget` | Max tokens allowed inside the thinking block |
| `--thinking-start-token` | Token that opens a thinking block (default: `<think>`) |
| `--thinking-end-token` | Token that closes a thinking block (default: `</think>`) |

When the budget is exceeded, the model is forced to emit `\n</think>` and transition to the answer. If `--enable-thinking` is passed but the model's chat template does not support it, the budget is applied only if the model generates the start token on its own.

### Speculative Decoding

Speed up generation by drafting several candidate tokens with a small "drafter" model and verifying them in a single target forward pass. Two drafter families are supported.

| Flag | Description |
|------|-------------|
| `--draft-model` | HuggingFace repo or local path for the drafter |
| `--draft-kind` | Drafter family — `dflash` (default) or `mtp` (Gemma 4) |
| `--draft-block-size` | Override the drafter's configured block size |

See [docs/usage.md](docs/usage.md) for Python API examples including batch generation.

#### DFlash (Qwen3.5)

A lightweight block-diffusion drafter that predicts multiple tokens per round, typically 2–3× faster.

```sh
# Text generation with speculative decoding
mlx_vlm.generate --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0 --enable-thinking

# Also works with images
mlx_vlm.generate --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --image examples/images/cats.jpg \
  --prompt "Describe this image." \
  --max-tokens 256 --temperature 0 --enable-thinking

# Server with speculative decoding
mlx_vlm.server --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash
```

#### Gemma 4 MTP

[Multi-Token Prediction](https://ai.google.dev/gemma/docs/mtp/mtp): Google's 4-layer "assistant" drafter that shares K/V with the target and drafts multiple tokens autoregressively from a constant position. Pass `--draft-kind mtp` to dispatch the MTP round-loop.

```sh
mlx_vlm.generate --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
  --draft-kind mtp --draft-block-size 4 \
  --prompt "Explain speculative decoding in 3 sentences." \
  --max-tokens 256 --temperature 0

# Server
mlx_vlm.server --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
  --draft-kind mtp --draft-block-size 4
```

Supported pairings (target ↔ drafter):

| Target                          | Drafter                                  |
|---------------------------------|------------------------------------------|
| `mlx-community/gemma-4-E2B-it-bf16`         | `mlx-community/gemma-4-E2B-it-assistant-bf16`        |
| `mlx-community/gemma-4-E4B-it-bf16`         | `mlx-community/gemma-4-E4B-it-assistant-bf16`        |
| `mlx-community/gemma-4-26B-A4B-it-bf16`     | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`    |
| `mlx-community/gemma-4-31B-it-bf16`         | `mlx-community/gemma-4-31B-it-assistant-bf16`        |

Measured speedups (greedy, byte-identical output): up to **3.94×** on 26B-A4B and **2.29×** on 31B at B=4. See [`mlx_vlm/speculative/drafters/gemma4_assistant/README.md`](mlx_vlm/speculative/drafters/gemma4_assistant/README.md) for full sweeps and architecture notes.

### Chat UI with Gradio

Launch a chat interface using Gradio:

```sh
mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python Script

Here's an example of how to use MLX-VLM in a Python script:

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
# image = [Image.open("...")] can also be used with PIL.Image.Image objects
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

#### Audio Example

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load model with audio support
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# Prepare audio input
audio = ["/path/to/audio1.wav", "/path/to/audio2.mp3"]
prompt = "Describe what you hear in these audio files."

# Apply chat template with audio
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_audios=len(audio)
)

# Generate output with audio
output = generate(model, processor, formatted_prompt, audio=audio, verbose=False)
print(output)
```

#### Multi-Modal Example (Image + Audio)

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load multi-modal model
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# Prepare inputs
image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = ""

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt,
    num_images=len(image),
    num_audios=len(audio)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, audio=audio, verbose=False)
print(output)
```

### Server (FastAPI)

Start the server:
```sh
mlx_vlm.server --port 8080

# Preload a model at startup (Hugging Face repo or local path)
mlx_vlm.server --model <hf_repo_or_local_path>

# Preload a model with adapter
mlx_vlm.server --model <hf_repo_or_local_path> --adapter-path <adapter_path>

# With trust remote code enabled (required for some models)
mlx_vlm.server --trust-remote-code
```

#### Server Options

- `--model`: Preload a model at server startup, accepts a Hugging Face repo ID or local path (optional, loads lazily on first request if omitted)
- `--adapter-path`: Path for adapter weights to use with the preloaded model
- `--draft-model`: Speculative drafter path or HF id (e.g. `z-lab/Qwen3.5-4B-DFlash`, `google/gemma-4-31B-it-assistant`) — enables speculative decoding for ~2× or higher throughput
- `--draft-kind`: Drafter family — `dflash` (default) or `mtp` (Gemma 4)
- `--draft-block-size`: Override the drafter's configured block size
- `--host`: Host address (default: `0.0.0.0`)
- `--port`: Port number (default: `8080`)
- `--trust-remote-code`: Trust remote code when loading models from Hugging Face Hub
- `--kv-bits`: Number of bits for KV cache quantization (e.g. `8` for uniform, `3.5` for TurboQuant)
- `--kv-quant-scheme`: KV cache quantization backend (`uniform` or `turboquant`)
- `--kv-group-size`: Group size for uniform KV cache quantization (default: `64`)
- `--max-kv-size`: Maximum KV cache size in tokens
- `--vision-cache-size`: Max number of cached vision features (default: `20`)
- `--log-level`: Logging level — `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)

You can also set trust remote code via environment variable:
```sh
MLX_TRUST_REMOTE_CODE=true mlx_vlm.server
```

The server provides multiple endpoints for different use cases and supports dynamic model loading/unloading with caching (one model at a time).

### Continuous Batching

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

### Automatic Prefix Caching (APC)

Automatic Prefix Caching reuses block-level K/V cache state across requests that share the same prefix. It is useful for repeated long documents, long chat histories, or retrieval contexts where each request appends a short new suffix.

APC has two tiers:

- **Warm memory**: keeps reusable `APCBlock` tensors in process memory. This is the fastest path, but it keeps both the reusable block pool and the runtime `KVCache`.
- **Warm disk**: persists cached prefixes as safetensors shards so they survive process restarts. Warm-disk reads build the layer-major prompt cache directly without promoting restored blocks into the `APCBlock` pool; writes can still populate both memory and disk tiers.

#### Python Script

Use `APCManager` directly when calling `stream_generate`:

```python
from pathlib import Path

from mlx_vlm import load, stream_generate
from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.prompt_utils import apply_chat_template

model_id = "Qwen/Qwen3-VL-4B-Instruct"
model, processor = load(model_id)

disk = DiskBlockStore(
    Path("~/.cache/mlx-vlm/caching").expanduser(),
    namespace=model_id,
    max_bytes=3 * (1 << 30),  # 3 GB disk cap; use None for uncapped
)
apc = APCManager(num_blocks=4096, block_size=16, disk=disk)

document = Path("long_document.txt").read_text()

try:
    # First request computes the full prefix and stores reusable K/V blocks.
    prompt1 = apply_chat_template(
        processor,
        model.config,
        prompt=f"{document}\n\nSummarize the key decisions.",
        num_images=0,
    )
    for _ in stream_generate(
        model, processor, prompt1, max_tokens=128, temperature=0.0, apc_manager=apc
    ):
        pass

    # Second request shares the same document prefix and only prefills the suffix.
    prompt2 = apply_chat_template(
        processor,
        model.config,
        prompt=f"{document}\n\nList the open engineering risks.",
        num_images=0,
    )
    for chunk in stream_generate(
        model, processor, prompt2, max_tokens=128, temperature=0.0, apc_manager=apc
    ):
        print(chunk.text, end="", flush=True)

    print(apc.stats_snapshot())
finally:
    apc.close()
```

To compare cold, warm-memory, and warm-disk behavior with a model:

```sh
python scripts/bench_apc_context_sweep.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --contexts 8000 20000 50000 100000 \
  --disk-cap-gb 0 \
  --shard-max-blocks 256
```

For a disk-eviction workload:

```sh
python scripts/bench_apc_disk_genstep.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --test-prompt-tokens 8000 \
  --fill-prompts 80 \
  --disk-cap-gb 3.0
```

#### Server

Enable in-memory APC for the server with environment variables:

```sh
APC_ENABLED=1 \
APC_NUM_BLOCKS=4096 \
mlx_vlm.server --model Qwen/Qwen3-VL-4B-Instruct --port 8080
```

Enable the persistent disk tier:

```sh
APC_ENABLED=1 \
APC_NUM_BLOCKS=4096 \
APC_DISK_PATH=~/.cache/mlx-vlm/caching \
APC_DISK_MAX_GB=3 \
APC_DISK_SHARD_MAX_BLOCKS=256 \
mlx_vlm.server --model Qwen/Qwen3-VL-4B-Instruct --port 8080
```

Repeated requests with the same long prefix will hit APC automatically:

```sh
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-APC-Tenant: demo" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [{
      "role": "user",
      "content": "Paste a long shared document here.\n\nNow answer question A."
    }],
    "max_tokens": 128
  }'
```

Use the same `X-APC-Tenant` value for requests that may share cached prefixes. Use different tenant values to isolate cache entries between users or workspaces.

Inspect and reset APC state:

```sh
curl http://localhost:8080/v1/cache/stats
curl -X POST http://localhost:8080/v1/cache/reset
```

Common APC environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APC_ENABLED` | `0` | Set to `1` to enable APC |
| `APC_NUM_BLOCKS` | `2048` | Number of in-memory APC blocks |
| `APC_BLOCK_SIZE` | `16` | Tokens per APC block |
| `APC_DISK_PATH` | unset | Directory for persistent disk shards |
| `APC_DISK_MAX_GB` | `0` | Disk cap in GB; `0` means uncapped |
| `APC_DISK_SHARD_MAX_BLOCKS` | `256` | Max blocks per disk segment shard |
| `APC_MAX_POOL_TENSORS` | `450000` | Stops adding memory blocks before the Metal resource limit; disk writes continue |
| `APC_LAYER_MAJOR_MEMORY_MIN_TOKENS` | `50000` | Store long warm-memory prefixes as compact layer-major snapshots instead of per-block tensors |
| `APC_HASH` | `fast` | Set to `sha256` for a stable cryptographic hash |

APC is disabled automatically for models that use a custom cache layout. On the server, APC is also skipped when KV-cache quantization is enabled.

#### KV Cache Quantization

Reduce KV cache memory during continuous batching with `--kv-bits`. Both uniform quantization and TurboQuant are supported:

```sh
# Uniform 8-bit KV cache quantization
mlx_vlm.server --model google/gemma-4-26b-a4b-it --kv-bits 8

# TurboQuant 3.5-bit (3-bit keys + 4-bit values)
mlx_vlm.server --model google/gemma-4-26b-a4b-it --kv-bits 3.5 --kv-quant-scheme turboquant
```

Full-attention layers use quantized batch caches while sliding-window layers keep their fixed-size rotating caches. The last full-attention layer stays unquantized (sensitive in deep models).

Tested with gemma-4-26b-a4b-it at 20K context:

| Config | Gen tok/s | KV Cache | KV Reduction |
|--------|-----------|----------|--------------|
| No quant | 50.3 | 0.624 GB | 1x |
| Uniform 8-bit | 52.6 | 0.469 GB | **1.33x** |
| TurboQuant 3.5-bit | 25.6 | 0.365 GB | **1.71x** |

> Models with all full-attention layers (e.g. Qwen, LLaMA) see larger reductions — up to 3.6x at 8-bit and 6.4x at 4-bit.

#### Log Probabilities

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

#### Structured Outputs

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

#### How It Works

- A dedicated generation thread runs a `BatchGenerator` that processes multiple requests in parallel
- Image requests are prefilled individually with their own vision embeddings, then join the shared decoding batch
- Text-only requests are batched together for efficient prefill
- After prefill, all requests decode together in a single batch, sharing GPU compute

#### Available Endpoints

- `/models` and `/v1/models` - List models available locally
- `/chat/completions` and `/v1/chat/completions` - OpenAI-compatible chat-style interaction endpoint with support for images, audio, and text
- `/responses` and `/v1/responses` - OpenAI-compatible responses endpoint
- `/health` - Check server status
- `/unload` - Unload current model from memory

#### Usage Examples

##### List available models

```sh
curl "http://localhost:8080/models"
```

##### Text Input

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

##### Image Input

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

##### Audio Support (New)
```sh
curl -X POST "http://localhost:8080/generate" \
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

##### Multi-Modal (Image + Audio)
```sh
curl -X POST "http://localhost:8080/generate" \
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

##### Responses Endpoint
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

#### Request Parameters

- `model`: Model identifier (required)
- `messages`: Chat messages for chat/OpenAI endpoints
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter
- `top_k`: Top-k sampling cutoff
- `min_p`: Min-p sampling threshold
- `repetition_penalty`: Penalty applied to repeated tokens
- `stream`: Enable streaming responses


## Activation Quantization (CUDA)

When running on NVIDIA GPUs with MLX CUDA, models quantized with `mxfp8` or `nvfp4` modes require activation quantization to work properly. This converts `QuantizedLinear` layers to `QQLinear` layers which quantize both weights and activations.

### Command Line

Use the `-qa` or `--quantize-activations` flag:

```sh
mlx_vlm.generate --model /path/to/mxfp8-model --prompt "Describe this image" --image /path/to/image.jpg -qa
```

### Python API

Pass `quantize_activations=True` to the `load` function:

```python
from mlx_vlm import load, generate

# Load with activation quantization enabled
model, processor = load(
    "path/to/mxfp8-quantized-model",
    quantize_activations=True
)

# Generate as usual
output = generate(model, processor, "Describe this image", image=["image.jpg"])
```

### Supported Quantization Modes

- `mxfp8` - 8-bit MX floating point
- `nvfp4` - 4-bit NVIDIA floating point

> **Note**: This feature is required for mxfp/nvfp quantized models on CUDA. On Apple Silicon (Metal), these models work without the flag.

## Multi-Image Chat Support

MLX-VLM supports analyzing multiple images simultaneously with select models. This feature enables more complex visual reasoning tasks and comprehensive analysis across multiple images in a single conversation.


### Usage Examples

#### Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = model.config

images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(images)
)

output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)
```

#### Command Line

```sh
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Compare these images" --image path/to/image1.jpg path/to/image2.jpg
```

## Video Understanding

MLX-VLM also supports video analysis such as captioning, summarization, and more, with select models.

### Supported Models

The following models support video chat:

1. Qwen2-VL
2. Qwen2.5-VL
3. Idefics3
4. LLaVA

With more coming soon.

### Usage Examples

#### Command Line
```sh
mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --max-pixels 224 224 --fps 1.0
```


These examples demonstrate how to use multiple images with MLX-VLM for more complex visual reasoning tasks.

## Vision Feature Caching

In multi-turn conversations about an image, the vision encoder runs on every turn even though the image hasn't changed. `VisionFeatureCache` stores projected vision features in an LRU cache keyed by image path, so the expensive vision encoder is only called once per unique image.

### How It Works

1. **First turn (cache miss)** -- `encode_image()` runs the full vision pipeline (vision tower + projector), stores the result in the cache, and passes it to the language model.
2. **Subsequent turns (cache hit)** -- the cached features are passed directly via `cached_image_features`, skipping the vision encoder entirely.
3. **Image switch** -- when the image changes, it's a new cache key so features are computed and cached. Switching back to a previous image is a cache hit.

The cache holds up to 8 entries (configurable) and uses LRU eviction.

### CLI

All chat interfaces use `VisionFeatureCache` automatically:

```sh
# Gradio chat UI
python -m mlx_vlm.chat_ui --model google/gemma-4-26b-a4b-it

# Interactive chat with Rich UI (load images with /image command)
python -m mlx_vlm.chat --model google/gemma-4-26b-a4b-it

# Inline chat mode
python -m mlx_vlm.generate \
  --model google/gemma-4-26b-a4b-it \
  --image path/to/image.jpg \
  --chat \
  --max-tokens 200
```

### Python

```python
from mlx_vlm import load, stream_generate, VisionFeatureCache
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-26b-a4b-it")
cache = VisionFeatureCache()

image = "path/to/image.jpg"

# Turn 1 -- cache miss, encodes image
prompt1 = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
for chunk in stream_generate(model, processor, prompt1, image=[image],
                              max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")

# Turn 2 -- cache hit, skips vision encoder
prompt2 = apply_chat_template(processor, model.config, "What colors do you see?", num_images=1)
for chunk in stream_generate(model, processor, prompt2, image=[image],
                              max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")
```

### Server

The server caches vision features automatically across requests for the same image. No configuration needed -- the cache is created when a model loads and cleared on unload.

```sh
mlx_vlm.server --model google/gemma-4-26b-a4b-it
```

Multi-turn conversations via `/v1/chat/completions` (streaming and non-streaming) and `/responses` all benefit. The same image sent across multiple requests will only be encoded once.

### Performance

Tested on `google/gemma-4-26b-a4b-it` over 10 multi-turn conversation turns:

| Metric | Without Cache | With Cache |
|--------|--------------|------------|
| Prompt TPS | ~48 | ~550-825 |
| Speedup | -- | **11x+** |
| Peak Memory | 52.66 GB | 52.66 GB (flat) |

Generation speed (~31 tok/s) and memory are unaffected -- only prompt processing gets faster.

## TurboQuant KV Cache

TurboQuant compresses the KV cache during generation, enabling longer context lengths with less memory while maintaining quality.

### Quick Start

```sh
# 3.5-bit KV cache quantization (3-bit keys + 4-bit values)
mlx_vlm generate \
  --model mlx-community/Qwen3.5-4B-4bit \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant \
  --prompt "Your long prompt here..."
```

```python
from mlx_vlm import generate

result = generate(
    model, processor, prompt,
    kv_bits=3.5,
    kv_quant_scheme="turboquant",
    max_tokens=256,
)
```

```sh
# Server with TurboQuant
mlx_vlm server \
  --model google/gemma-4-26b-a4b-it \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

### How It Works

TurboQuant uses random rotation + codebook quantization ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) to compress KV cache entries from 16-bit to 2-4 bits per dimension:

- **Keys & Values**: MSE codebook quantization with Hadamard rotation
- **Fractional bits** (e.g. 3.5): uses lower bits for keys, higher for values (3-bit K + 4-bit V)

Custom Metal kernels fuse score computation and value aggregation directly on packed quantized data, avoiding full dequantization during decode.

### Performance

Tested on Qwen3.5-4B-4bit at 128k context:

| Metric | Baseline | TurboQuant 3.5-bit |
|--------|----------|-------------------|
| KV Memory | 4.1 GB | 0.97 GB (**76% reduction**) |
| Peak Memory | 18.3 GB | 17.3 GB (**-1.0 GB**) |

At 512k+ contexts, TurboQuant's per-layer attention is **faster than FP16 SDPA** due to reduced memory bandwidth requirements.

Tested on gemma-4-31b-it at 128k context:

| Metric | Baseline | TurboQuant 3.5-bit |
|--------|----------|-------------------|
| KV Memory | 13.3 GB | 4.9 GB (**63% reduction**) |
| Peak Memory | 75.2 GB | 65.8 GB (**-9.4 GB**) |

### Supported Bit Widths

| Bits | Compression | Best For |
|------|------------|----------|
| 2 | ~8x | Maximum compression, some quality loss |
| 3 | ~5x | Good balance of quality and compression |
| 3.5 | ~4.5x | Recommended default (3-bit keys + 4-bit values) |
| 4 | ~4x | Best quality, moderate compression |

### Compatibility

TurboQuant automatically quantizes `KVCache` layers (global attention). Models with `RotatingKVCache` (sliding window) or `ArraysCache` (MLA/absorbed keys) keep their native cache format for those layers since they are already memory-efficient.

TurboQuant is supported in both single-request generation and continuous batching on the server. In continuous batching mode, KV states are stored in TurboQuant's compressed format and dequantized at attention time (custom Metal kernels are not yet batch-aware).

## Distributed Inference

mlx-vlm supports distributed inference across multiple computers. It works by sharding the language model (not the vision tower), because the LLM is much larger and vision embeddings only need to be computed once.

The parallel implementation is compatible with [mlx-lm](https://github.com/ml-explore/mlx-lm) sharding primitives.

See [docs/usage.md](https://github.com/Blaizzy/mlx-vlm/blob/main/docs/usage.md#distributed-inference) for command-line examples.

# Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA.

## LoRA & QLoRA

To learn more about LoRA, please refer to the [LoRA.md](./mlx_vlm/LORA.MD) file.
