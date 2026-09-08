# LLaVA-OneVision

SigLIP vision tower + Qwen2 language model, supporting single images, multiple
images, and video in one checkpoint.

## Checkpoints

| Model | Notes |
|-------|-------|
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | image + multi-image + video |
| `llava-hf/llava-onevision-qwen2-0.5b-si-hf` | single-image tuned |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | image + multi-image + video |
| `llava-hf/llava-onevision-qwen2-7b-si-hf` | single-image tuned |
| `llava-hf/llava-onevision-qwen2-7b-ov-chat-hf` | chat-tuned |
| `llava-hf/llava-onevision-qwen2-72b-ov-hf` | quantize before loading |

`lmms-lab/LLaVA-OneVision-1.5-*` is a different architecture (RICE-ViT vision
tower) and is not handled by this implementation.

## Usage

```sh
mlx_vlm.generate --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --prompt "What is in this image?" --image image.jpg --max-tokens 100

mlx_vlm.generate --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --prompt "What is happening in this video?" --video video.mp4 --fps 1.0
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
prompt = apply_chat_template(processor, model.config, "Compare these images.", num_images=2)
print(generate(model, processor, prompt, image=["a.jpg", "b.jpg"], max_tokens=100).text)
```

## Token budget

Images are tiled with AnyRes and packed as `base + unpadded grid + newlines`,
capped by `vision_aspect_ratio` (`anyres_max_9`). A 384px tower with patch 14
gives 729 base features, so a full-resolution image costs up to ~7.3k tokens; a
video costs `frames * 196 + 1`. Prefill dominates runtime for both.

## Preprocessing

The image processor `AutoImageProcessor` resolves for this model type is
torchvision-backed, so this implementation uses the PIL backend directly and
handles video frames with numpy — mlx-vlm needs no torch install. Resampling
differs slightly from the torchvision path (~1e-2 on normalized pixels), which
does not change greedy output.
