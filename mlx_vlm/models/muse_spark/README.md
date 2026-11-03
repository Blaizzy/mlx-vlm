# Muse Spark Open

Muse Spark supports text generation and image, video, and audio understanding
through the standard `mlx_vlm.load`, `generate`, and CLI APIs. The audio encoder
consumes mono 24 kHz waveforms directly and produces one token per 1,920 samples
(80 ms). Multiple clips and prompts mixing images, videos, and audio are supported.

Supported checkpoint layouts:

- [Meta's Muse-Spark-Open-816B-A42B-NVFP4](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4),
  including its vision and audio weights.
- MLX checkpoints converted by mlx-vlm or mlx-lm, including affine and NVFP4
  expert weights. Existing mlx-lm conversions that omit `audio_tower` weights
  support text, images, and video; audio requests require an audio-inclusive checkpoint.

## Image input

```sh
mlx_vlm.generate \
  --model mlx-community/Muse-Spark-Open-816B-A42B-4bit \
  --image photo.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Audio input

```sh
mlx_vlm.generate \
  --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --audio recording.wav \
  --prompt "Transcribe this recording." \
  --max-tokens 256
```

Audio files are downmixed and resampled by the shared input loader. NumPy arrays
must already be mono at 24 kHz. The processor right-pads waveforms to a multiple
of the encoder stride and tracks each clip's true length, so padding does not
insert additional audio tokens into the prompt.

## Video input

```sh
mlx_vlm.generate \
  --model mlx-community/Muse-Spark-Open-816B-A42B-4bit \
  --video clip.mp4 \
  --fps 2 \
  --prompt "Describe what happens in this video." \
  --max-tokens 256
```

The processor uniformly samples at 2 fps with a maximum of 600 frames by default.
It packs adjacent sampled frames into temporal patches, repeats the final frame
when needed, and inserts source timestamps into the prompt. Frames are resized
to at most 256 spatial tokens per temporal group. The vision encoder processes
each group independently, preserving the checkpoint's attention boundaries.
`--fps`, `--video-num-frames`, and `--video-max-frames` override sampling;
an explicit frame count takes precedence over fps. Multiple video paths are supported.

In Python, include the video when applying the chat template as well as when
generating:

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Muse-Spark-Open-816B-A42B-4bit")
prompt = apply_chat_template(
    processor, model.config, "Summarize this clip.", video="clip.mp4", num_images=0
)
result = generate(model, processor, prompt, video="clip.mp4", max_tokens=256)
```

Decoded video inputs may be RGB `uint8` arrays in `(T, C, H, W)` or
`(T, H, W, C)` layout. Supply their frame rate via `fps`, or pass source
`video_metadata` to preserve timestamps after external sampling. Video input
processes visual frames; supply a separate `audio` input to include the soundtrack.

## Python

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("meta-models/Muse-Spark-Open-816B-A42B-NVFP4")
prompt = apply_chat_template(
    processor,
    model.config,
    "Describe the picture and summarize the recording.",
    num_images=1,
    num_audios=1,
)
result = generate(
    model,
    processor,
    prompt,
    image="photo.jpg",
    audio="recording.wav",
    max_tokens=256,
)
print(result.text)
```

The model stops on `<|eot|>` / `<|end|>`, preserving `<|eom|>` transitions between
reasoning and answer messages. Completed text is cleaned to the final user
channel, following the existing Muse processor convention.
