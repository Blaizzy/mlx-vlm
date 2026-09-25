# Getting Started

## Installation

Install the package from PyPI:

```sh
pip install -U mlx-vlm
```

The [Gradio chat UI](#chat-ui-with-gradio) needs an extra dependency that is not part of the base install:

```sh
pip install -U 'mlx-vlm[ui]'
```

Quote the package name so that shells which expand square brackets, such as `zsh`, do not treat `[ui]` as a glob pattern.

### Optional extras

Install extra dependency groups with `pip install -U 'mlx-vlm[<extra>]'` (quote the name so shells like `zsh` don't expand the brackets). Combine them with commas, e.g. `pip install -U 'mlx-vlm[ui,train]'`.

| Extra | Adds | For |
|-------|------|-----|
| `ui` | `gradio` | The Gradio chat UI (`mlx_vlm.chat_ui`) |
| `train` | `datasets` (which pulls in `pandas`) | LoRA / QLoRA fine-tuning |
| `realtime` | `sounddevice` | Realtime full-duplex speech (server `/v1/realtime`) |
| `cuda` | `mlx-cuda` | Running on NVIDIA GPUs with MLX CUDA |
| `cpu` | `mlx-cpu` | CPU-only MLX builds |

## Command Line Interface (CLI)

Generate output from a model using the CLI:

```sh
# Text generation
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Hello, how are you?"

# Image
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg

# Audio
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you hear" --audio /path/to/audio.wav

# Image + Audio
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you see and hear" --image /path/to/image.jpg --audio /path/to/audio.wav
```

For every flag and subcommand, see the [CLI reference](cli.md).

## Chat UI with Gradio

The Gradio chat UI requires the optional `ui` extra:

```sh
pip install -U 'mlx-vlm[ui]'
```

Then launch the chat interface:

```sh
mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

## Python Script

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
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

### Audio

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

audio = ["/path/to/audio1.wav", "/path/to/audio2.mp3"]
prompt = "Describe what you hear in these audio files."

formatted_prompt = apply_chat_template(processor, config, prompt, num_audios=len(audio))
output = generate(model, processor, formatted_prompt, audio=audio, verbose=False)
print(output)
```

### Image + Audio

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = ""

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image), num_audios=len(audio)
)
output = generate(model, processor, formatted_prompt, image, audio=audio, verbose=False)
print(output)
```

## Multiple images

MLX-VLM can analyze several images at once with select models, for comparison and cross-image reasoning.

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = model.config

images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(images))
output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)
```

```sh
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Compare these images" --image path/to/image1.jpg path/to/image2.jpg
```

## Video understanding

MLX-VLM supports video analysis — captioning, summarization, and more — with select models (Qwen2-VL, Qwen2.5-VL, Idefics3, LLaVA, MiniMax M3, and more coming soon).

```sh
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --fps 1.0
```

## Vision feature caching

In multi-turn conversations about an image, the vision encoder otherwise runs every turn even though the image hasn't changed. `VisionFeatureCache` stores projected vision features in an LRU cache keyed by image path, so the encoder runs only once per unique image (default 8 entries, LRU eviction).

```python
from mlx_vlm import load, stream_generate, VisionFeatureCache
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-26b-a4b-it")
cache = VisionFeatureCache()
image = "path/to/image.jpg"

# Turn 1 -- cache miss, encodes the image
prompt1 = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
for chunk in stream_generate(model, processor, prompt1, image=[image], max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")

# Turn 2 -- cache hit, skips the vision encoder
prompt2 = apply_chat_template(processor, model.config, "What colors do you see?", num_images=1)
for chunk in stream_generate(model, processor, prompt2, image=[image], max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")
```

The server applies the same cache automatically across `/v1/chat/completions` and `/responses` requests for the same image — no configuration needed. On `google/gemma-4-26b-a4b-it` over 10 multi-turn turns, prompt throughput rose from ~48 to ~550–825 TPS (**11×+**) with flat peak memory; generation speed is unchanged.

## Speculative decoding

Speed up generation by drafting several candidate tokens with a small "drafter" model and verifying them in a single target forward pass (typically 2–3× faster).

```sh
mlx_vlm.generate --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt "Write a quicksort in Python." \
  --max-tokens 512 --temperature 0 --enable-thinking
```

See [Speculative Decoding](performance/speculative-decoding.md) for the drafter families (DFlash, MTP, EAGLE-3), the Python API, supported pairings, and measured speedups.

## Distributed inference

MLX-VLM supports distributed inference across multiple computers by sharding the language model (not the vision tower) — the LLM is much larger, and vision embeddings only need to be computed once. The parallel implementation is compatible with [mlx-lm](https://github.com/ml-explore/mlx-lm) sharding primitives.

For example, running Kimi K2.6 (a 1T-parameter model) across several machines (for a smaller option try `mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16`):

```sh
mlx.launch \
    --hostfile ring-thunderbolt.json \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    -- \
    mlx-vlm/examples/sharded_generate.py \
    --model moonshotai/Kimi-K2.6 \
    --prompt "Describe this image" \
    --image mlx-vlm/examples/images/scene_1.jpg
```

We recommend the JACCL protocol over Thunderbolt. See the [MLX distributed communication guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) for details.

## Examples

Runnable notebooks live in the [`examples/`](https://github.com/Blaizzy/mlx-vlm/tree/main/examples) directory:

- [multi_image_generation.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/multi_image_generation.ipynb)
- [object_detection.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/object_detection.ipynb)
- [object_pointing.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/object_pointing.ipynb)
- [ocr_with_region.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/ocr_with_region.ipynb)
- [text_extraction.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/text_extraction.ipynb)
- [video_understanding.ipynb](https://github.com/Blaizzy/mlx-vlm/blob/main/examples/video_understanding.ipynb)

Images and videos used by the notebooks are in `examples/images/` and `examples/videos/`.

## Next steps

- [Models](models.md) — the full catalog of supported model families.
- [Server](server.md) — the OpenAI-compatible FastAPI server.
- [Fine-tuning](fine-tuning.md) — LoRA and QLoRA.
- Performance: [Speculative Decoding](performance/speculative-decoding.md) · [Prefix Caching](performance/prefix-caching.md) · [Quantization](performance/quantization.md) · [KV Cache Quantization](performance/kv-cache-quantization.md).
