[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (VLMs with audio and video support) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
- [Continuous Batching](#continuous-batching)
- [Multi-Image Chat Support](#multi-image-chat-support)
  - [Supported Models](#supported-models)
  - [Usage Examples](#usage-examples)
- [Model-Specific Documentation](#model-specific-documentation)
- [Fine-tuning](#fine-tuning)

## Model-Specific Documentation

Some models have detailed documentation with prompt formats, examples, and best practices:

| Model | Documentation |
|-------|---------------|
| DeepSeek-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/README.md) |
| DeepSeek-OCR-2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md) |

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

# With trust remote code enabled (required for some models)
mlx_vlm.server --trust-remote-code
```

#### Server Options

- `--host`: Host address (default: `0.0.0.0`)
- `--port`: Port number (default: `8080`)
- `--model`: Pre-load a model at startup for faster first request
- `--adapter-path`: Adapter weights to load with the model
- `--trust-remote-code`: Trust remote code when loading models from Hugging Face Hub
- `--vision-cache-size`: Max number of cached vision features (default: `20`)

```sh
# Pre-load model with custom vision cache
mlx_vlm.server --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit --vision-cache-size 50
```

**Vision Feature Caching**: The server caches vision encoder outputs so repeated images skip the vision tower entirely. On gemma4, this saves ~230ms and ~1GB peak memory per cached image hit. The cache uses LRU eviction and is cleared automatically on model unload. Set `--vision-cache-size 0` to disable.

You can also set trust remote code via environment variable:
```sh
MLX_TRUST_REMOTE_CODE=true mlx_vlm.server
```

The server provides multiple endpoints for different use cases and supports dynamic model loading/unloading with caching (one model at a time).

#### Available Endpoints

- `/models` - List models available locally
- `/chat/completions` - OpenAI-compatible chat-style interaction endpoint with support for images, audio, and text
- `/responses` - OpenAI-compatible responses endpoint
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
        "content": "Hello, how are you",
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
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "This is today'\''s chart for energy demand in California. Can you provide an analysis of the chart and comment on the implications for renewable energy in California?"
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
          {"type": "input_audio", "input_audio": "/path/to/audio1.wav"}
          {"type": "input_audio", "input_audio": "https://example.com/audio2.mp3"}
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
- `stream`: Enable streaming responses


## Continuous Batching

The server supports continuous batching for higher throughput when handling multiple concurrent requests. New requests join the active batch immediately without waiting for existing requests to finish, and mixed batches of image and text-only requests are supported.

Continuous batching is enabled automatically when the server loads a model. You can pre-load a model at startup so it's ready to serve immediately:

```sh
mlx_vlm.server --port 8080 --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

Verify via the health endpoint:

```sh
curl http://localhost:8080/health
# {"status":"healthy","loaded_model":"...","continuous_batching_enabled":true}
```

If `--model` is omitted, the model is loaded on the first request.

### How It Works

- A dedicated generation thread runs a `BatchGenerator` that processes multiple requests in parallel
- Image requests are prefilled individually with their own vision embeddings, then join the shared decoding batch
- Text-only requests are batched together for efficient prefill
- After prefill, all requests decode together in a single batch, sharing GPU compute

### Python Example

You can also use continuous batching directly via the `ResponseGenerator`:

```python
from mlx_vlm.server import ResponseGenerator, GenerationArguments
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
config = model.config

# Get stop tokens
stop_tokens = set()
if isinstance(config.eos_token_id, list):
    stop_tokens.update(config.eos_token_id)
elif config.eos_token_id is not None:
    stop_tokens.add(config.eos_token_id)

rg = ResponseGenerator(model=model, processor=processor, stop_tokens=stop_tokens)
args = GenerationArguments(max_tokens=100, temperature=0.0)

# Submit a request (text-only or with images)
prompt = apply_chat_template(processor, config, "What is in this image?", num_images=1)
ctx, token_iter = rg.generate(
    prompt=prompt, images=["path/to/image.jpg"], args=args
)

# Stream tokens
for token in token_iter:
    print(token.text, end="", flush=True)
    if token.finish_reason:
        break

rg.stop_and_join()
```

### Concurrent Requests

Multiple requests can be submitted from different threads. They will be batched together automatically:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def submit(name, prompt, images=None):
    formatted = apply_chat_template(processor, config, prompt, num_images=len(images) if images else 0)
    ctx, it = rg.generate(prompt=formatted, images=images, args=args)
    text = "".join(t.text for t in it)
    return name, text

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(submit, "text", "What is 2+2?"),
        pool.submit(submit, "image", "Describe this image.", ["photo.jpg"]),
        pool.submit(submit, "text2", "Capital of France?"),
    ]
    for f in as_completed(futures):
        name, text = f.result()
        print(f"[{name}]: {text}")
```

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

# Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA.

## LoRA & QLoRA

To learn more about LoRA, please refer to the [LoRA.md](./mlx_vlm/LORA.MD) file.
