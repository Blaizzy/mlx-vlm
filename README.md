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

| Flag | Description |
|------|-------------|
| `--draft-model` | HuggingFace repo or local path for the drafter |
| `--draft-kind` | Drafter family — `dflash` (default), `eagle3`, or `mtp` (native/assistant MTP) |
| `--draft-block-size` | Override the drafter's configured block size |

See [docs/usage.md](docs/usage.md) for Python API examples including batch generation.

#### DFlash (Qwen3.5 and Muse Glimmer)

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

Muse Glimmer's published assistant checkpoint is auto-detected as DFlash:

```sh
mlx_vlm.generate --model meta-models/Muse-Glimmer-30B \
  --draft-model meta-models/Muse-Glimmer-30B-assistant \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0

mlx_vlm.server --model meta-models/Muse-Glimmer-30B \
  --draft-model meta-models/Muse-Glimmer-30B-assistant
```

DFlash draft-cache windowing is available from the Python API. During
speculative decoding the target model still verifies every proposed token with
its full KV cache; this knob only changes the DFlash drafter cache. When
`draft_window_size` is set, the drafter keeps at most that many recent committed
tokens in its own KV cache instead of attending over the full generated prefix.
That reduces draft-side cache length and memory, but it can lower acceptance
because the drafter has less context than the target verifier. On MLX, the full
draft cache is usually faster for Qwen3.5 DFlash, so windowing defaults to
`None`; set it only when you want to experiment with this compact recent-token
cache tradeoff:

```python
from mlx_vlm import load
from mlx_vlm.generate import generate
from mlx_vlm.speculative.drafters import load_drafter

model, processor = load("Qwen/Qwen3.5-4B")
draft_model, draft_kind = load_drafter("z-lab/Qwen3.5-4B-DFlash")
draft_model.config.draft_window_size = 256  # None disables windowing

result = generate(
    model,
    processor,
    "Write a quicksort in Python.",
    max_tokens=512,
    temperature=0,
    draft_model=draft_model,
    draft_kind=draft_kind,
)
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

#### Gemma 4 EAGLE-3

[EAGLE-3](https://sgl-project.github.io/SpecForge/concepts/EAGLE3.html) drafts from three target hidden-state captures with a lightweight one-layer speculator. The Red Hat Speculators checkpoint auto-detects as `--draft-kind eagle3`.

```sh
mlx_vlm.generate --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model RedHatAI/gemma-4-31B-it-speculator.eagle3 \
  --prompt "Explain speculative decoding in 3 sentences." \
  --max-tokens 256 --temperature 0

# Server
mlx_vlm.server --model mlx-community/gemma-4-31B-it-bf16 \
  --draft-model RedHatAI/gemma-4-31B-it-speculator.eagle3
```

#### MiniMax M3 EAGLE-3

MiniMax M3 supports the released `Inferact/MiniMax-M3-EAGLE3` drafter. Convert
the target with `mlx_vlm.convert` because `mlx_lm.convert` does not know the
`minimax_m3_vl` model type.

```sh
mlx_vlm.convert \
  --hf-path MiniMaxAI/MiniMax-M3 \
  --mlx-path ~/MiniMax-M3-4bit \
  --quantize --q-bits 4 \
  --trust-remote-code

mlx_vlm.convert \
  --hf-path Inferact/MiniMax-M3-EAGLE3 \
  --mlx-path ~/MiniMax-M3-EAGLE3

mlx_vlm.generate \
  --model ~/MiniMax-M3-4bit \
  --draft-model ~/MiniMax-M3-EAGLE3 \
  --draft-kind eagle3 \
  --draft-block-size 3 \
  --prompt "Explain MiniMax Sparse Attention in one paragraph." \
  --max-tokens 256 --temperature 0
```

The public MiniMax M3 BF16 checkpoint advertises MTP metadata but does not
publish `mtp` or `nextn` tensors, so use the released EAGLE-3 drafter for that
checkpoint.

MiniMax M3 also supports image/video prompts, MiniMax thinking tags, MiniMax
tool-call parsing, MSA index caches, and MXFP8 config loading. See
[`mlx_vlm/models/minimax_m3_vl/README.md`](mlx_vlm/models/minimax_m3_vl/README.md)
for model-specific conversion and runtime notes.

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

MLX-VLM can load existing affine 1-bit MLX checkpoints without a custom MLX
build. When a checkpoint declares `"bits": 1`, compatible `Linear` and
`Embedding` layers are replaced automatically with an inference-only module
that JIT-compiles its Metal kernel from Python.

The checkpoint must use MLX's packed `uint32` weight layout, include `scales`
and `biases`, and declare a group size of `32`, `64`, or `128`:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 1,
    "mode": "affine"
  }
}
```

Load and generate normally; no extra inference flag is needed:

```python
from mlx_vlm import generate, load

model, processor = load("path/to/1bit-model")
result = generate(model, processor, "Describe this image", image=["image.jpg"])
```

This path is for inference from an already quantized checkpoint. Converting a
floating-point model to 1-bit still requires a quantizer that can produce the
packed weights and affine parameters.

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
5. MiniMax M3

With more coming soon.

### Usage Examples

#### Command Line
```sh
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --fps 1.0
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

## Distributed Inference

mlx-vlm supports distributed inference across multiple computers. It works by sharding the language model (not the vision tower), because the LLM is much larger and vision embeddings only need to be computed once.

The parallel implementation is compatible with [mlx-lm](https://github.com/ml-explore/mlx-lm) sharding primitives.

See [docs/usage.md](https://github.com/Blaizzy/mlx-vlm/blob/main/docs/usage.md#distributed-inference) for command-line examples.

## Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA. Fine-tuning (and the
eval scripts) need the training extra, which is not installed by default:

```bash
pip install "mlx-vlm[train]"
```

## LoRA & QLoRA

To learn more about LoRA, please refer to the [LoRA.md](./mlx_vlm/LORA.MD) file.
