[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models (VLMs with audio and video support) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Agent Skills](#agent-skills)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Thinking Budget](#thinking-budget)
  - [Speculative Decoding](#speculative-decoding)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
- [Server (FastAPI)](#server-fastapi)
- [1-bit Affine Inference](#1-bit-affine-inference)
- [Activation Quantization (CUDA)](#activation-quantization-cuda)
- [Multi-Image Chat Support](#multi-image-chat-support)
- [Video Understanding](#video-understanding)
- [Vision Feature Caching](#vision-feature-caching)
- [Model-Specific Documentation](#model-specific-documentation)
- [Distributed Inference](#distributed-inference)
- [Fine-tuning](#fine-tuning)

Full reference docs live in [`docs/`](docs/) (also published at
[Blaizzy.github.io/mlx-vlm](https://Blaizzy.github.io/mlx-vlm)):
[Server](docs/server.md) · [Prefix caching](docs/prefix-caching.md) ·
[KV cache quantization](docs/kv-cache-quantization.md) · [Usage](docs/usage.md) ·
[CLI reference](docs/cli_reference.md).

## Model-Specific Documentation

Some models have detailed documentation with prompt formats, examples, and best practices:

| Model | Documentation |
|-------|---------------|
| DeepSeek-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/README.md) |
| DeepSeek-OCR-2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md) |
| Unlimited-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/unlimited_ocr/README.md) |
| DOTS-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| DOTS-MOCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| ERNIE 4.5 VL | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/README.md) |
| GLM-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm_ocr/README.md) |
| Phi-4 Reasoning Vision | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4_siglip/README.md) |
| MiniCPM-o | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/README.md) |
| PaddleOCR-VL | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/README.md) |
| Phi-4 Multimodal | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4mm/README.md) |
| MolmoPoint | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/README.md) |
| LocateAnything | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/README.md) |
| Moondream2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/moondream2/README.md) |
| Moondream3 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/moondream3/README.md) |
| Gemma 4 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md) |
| MiniMax M3 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/README.md) |
| Falcon-OCR | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/falcon_ocr/README.md) |
| Granite Vision 3.2 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite_vision/README.md) |
| Granite 4.0 Vision | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite4_vision/README.md) |
| MiniCPM-V 4.6 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_6/README.md) |
| GLiNER2.5 | [Docs](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gliner2_5/README.md) |

## Installation

The easiest way to get started is to install the `mlx-vlm` package using pip:

```sh
pip install -U mlx-vlm
```

The [Gradio chat UI](#chat-ui-with-gradio) needs an extra dependency that is not
part of the base install:

```sh
pip install -U 'mlx-vlm[ui]'
```

Quote the package name so that shells which expand square brackets, such as
`zsh`, do not treat `[ui]` as a glob pattern.

## Agent Skills

This repo ships an agent-skills bundle under `skills/` for common MLX-VLM workflows — usage, conversion, development, and support. Skills load into a coding agent (Claude Code, Codex, Gemini) so it follows the right project conventions instead of guessing.

| Skill | Description |
|-------|-------------|
| `cli-inference` | Run and debug command-line inference (`mlx_vlm.generate`) — text/image/audio inputs and image-generation flags. |
| `server-inference` | Run and debug the local server across the models, chat, responses, messages, audio, image, cache, and metrics endpoints. |
| `convert-quantize` | Convert and quantize Hugging Face models to MLX (`mlx_vlm.convert`) — bits/group size, quant modes, RTN/AWQ, mixed recipes. |
| `add-new-model` | Port a new architecture into `mlx_vlm/models` — config, weight-name mapping, reuse a similar model, add a test class. |
| `benchmarking` | Produce credible, reproducible perf numbers and fork-vs-main A/B tables for PRs. |
| `contributing` | Shape a change to pass review — code/config/test placement, pre-commit hooks, and PR expectations. |
| `hf-cache-models` | List MLX-VLM-supported (and, with `--check-arch`, loadable) models in the local Hugging Face cache. |
| `reproducible-github-issues` | Turn CLI or server failures into concise, reproducible GitHub issues. |

Validate the bundle at any time:

```sh
python3 skills/scripts/validate_skills.py
```

Install from a local checkout:

```sh
# Claude Code
/plugin marketplace add /path/to/mlx-vlm
/plugin install mlx-vlm-skills@mlx-vlm

# Codex CLI
codex plugin marketplace add /path/to/mlx-vlm
codex plugin add mlx-vlm-skills@mlx-vlm

# Gemini CLI
gemini extensions install /path/to/mlx-vlm/skills
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

On the server, thinking mode is disabled by default. Start the server with `--enable-thinking` to make thinking mode the default for requests that do not specify it:

```sh
mlx_vlm.server --model Qwen/Qwen3.5-4B --enable-thinking
```

You can also set server defaults for the thinking budget and delimiter tokens:

```sh
mlx_vlm.server --model Qwen/Qwen3.5-4B \
  --enable-thinking \
  --thinking-budget 512 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>"
```

Requests can override the server defaults with `enable_thinking`, `thinking_budget`, `thinking_start_token`, or `thinking_end_token`.

### Speculative Decoding

Speed up generation by drafting several candidate tokens with a small "drafter" model and verifying them in a single target forward pass. Three drafter families are supported.

See [Speculative Decoding](docs/speculative-decoding.md) for CLI/server flags, per-family setup (DFlash, Gemma 4 MTP, Gemma 4 EAGLE-3, MiniMax M3 EAGLE-3), supported pairings, and measured speedups.
### Chat UI with Gradio

The Gradio chat UI requires the optional `ui` extra, which the base `mlx-vlm`
install does not include:

```sh
pip install -U 'mlx-vlm[ui]'
```

Then launch the chat interface:

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

## Server (FastAPI)

MLX-VLM ships an OpenAI-compatible FastAPI server with continuous batching and
streaming, plus endpoints for chat, responses, embeddings, reranking, audio
(TTS/STT), image generation, and realtime speech.

```sh
# Start (optionally preload a model; it otherwise loads on first request)
mlx_vlm.server --port 8080 --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

```sh
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen2.5-VL-3B-Instruct-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

See the full reference for CLI flags, every endpoint, streaming, log-probs, and
structured outputs:

- **[Server reference](docs/server.md)** — all flags/endpoints, curl examples, live `/v1/settings`.
- **[Automatic Prefix Caching](docs/prefix-caching.md)** — reuse K/V across requests that share a prefix.
- **[KV cache quantization](docs/kv-cache-quantization.md)** — `--kv-bits`, TurboQuant, per-tensor schemes.

## 1-bit Affine Inference

MLX-VLM can load existing affine 1-bit MLX checkpoints without a custom MLX build.

See [Quantization](docs/quantization.md#1-bit-affine-inference) for the checkpoint layout and load/generate examples.

## Activation Quantization (CUDA)

When running on NVIDIA GPUs with MLX CUDA, models quantized with `mxfp8` or `nvfp4` modes require activation quantization to work properly.

See [Quantization](docs/quantization.md#activation-quantization-cuda) for the CLI flag, Python API, and supported modes.
## Multi-Image Chat Support

MLX-VLM supports analyzing multiple images simultaneously with select models. This feature enables more complex visual reasoning tasks and comprehensive analysis across multiple images in a single conversation.

See [Multimodal & Vision Features](docs/multimodal.md#multi-image-chat-support) for examples.

## Video Understanding

MLX-VLM also supports video analysis such as captioning, summarization, and more, with select models.

See [Multimodal & Vision Features](docs/multimodal.md#video-understanding) for supported models and examples.

## Vision Feature Caching

In multi-turn conversations about an image, the vision encoder runs on every turn even though the image hasn't changed. `VisionFeatureCache` stores projected vision features in an LRU cache keyed by image path, so the expensive vision encoder is only called once per unique image.

See [Multimodal & Vision Features](docs/multimodal.md#vision-feature-caching) for how it works, the CLI/Python/server paths, and benchmarks.

## Distributed Inference

mlx-vlm supports distributed inference across multiple computers. It works by sharding the language model (not the vision tower), because the LLM is much larger and vision embeddings only need to be computed once.

The parallel implementation is compatible with [mlx-lm](https://github.com/ml-explore/mlx-lm) sharding primitives.

See [docs/usage.md](https://github.com/Blaizzy/mlx-vlm/blob/main/docs/usage.md#distributed-inference) for command-line examples.


## Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA.

See [Fine-tuning](docs/fine-tuning.md) for the `mlx-vlm[train]` extra and LoRA/QLoRA usage.
