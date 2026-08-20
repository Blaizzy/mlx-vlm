# Multimodal & Vision Features

Working with multiple images, video, and cached vision features.

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

