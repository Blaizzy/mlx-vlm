[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (VLMs with audio and video support) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
- [Multi-Image Chat Support](#multi-image-chat-support)
  - [Supported Models](#supported-models)
  - [Usage Examples](#usage-examples)
- [Fine-tuning](#fine-tuning)

## Installation

The easiest way to get started is to install the `mlx-vlm` package using pip:

```sh
pip install -U mlx-vlm
```

## Usage

### Command Line Interface (CLI)

Generate output from a model using the CLI:

```sh
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
mlx_vlm.server
```

The server provides multiple endpoints for different use cases and supports dynamic model loading/unloading with caching (one model at a time).

#### Available Endpoints

- `/generate` - Main generation endpoint with support for images, audio, and text
- `/chat` - Chat-style interaction endpoint
- `/responses` - OpenAI-compatible endpoint
- `/health` - Check server status
- `/unload` - Unload current model from memory

#### Usage Examples

##### Basic Image Generation
```sh
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
    "image": ["/path/to/repo/examples/images/renewables_california.png"],
    "prompt": "This is today'\''s chart for energy demand in California. Can you provide an analysis of the chart and comment on the implications for renewable energy in California?",
    "system": "You are a helpful assistant.",
    "stream": true,
    "max_tokens": 1000
  }'
```

##### Audio Support (New)
```sh
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "audio": ["/path/to/audio1.wav", "https://example.com/audio2.mp3"],
    "prompt": "Describe what you hear in these audio files",
    "stream": true,
    "max_tokens": 500
  }'
```

##### Multi-Modal (Image + Audio)
```sh
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "image": ["/path/to/image.jpg"],
    "audio": ["/path/to/audio.wav"],
    "prompt": "",
    "max_tokens": 1000
  }'
```

##### Chat Endpoint
```sh
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": "What is in this image?",
        "images": ["/path/to/image.jpg"]
      }
    ],
    "max_tokens": 100
  }'
```

##### OpenAI-Compatible Endpoint
```sh
curl -X POST "http://localhost:8000/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is in this image?"},
          {"type": "input_image", "image": "/path/to/image.jpg"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

#### Request Parameters

- `model`: Model identifier (required)
- `prompt`: Text prompt for generation
- `image`: List of image URLs or local paths (optional)
- `audio`: List of audio URLs or local paths (optional, new)
- `system`: System prompt (optional)
- `messages`: Chat messages for chat/OpenAI endpoints
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter
- `stream`: Enable streaming responses


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
