# MiniMax-H3

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) generates 24 FPS
video with synchronized 32 kHz stereo audio. Three workflows are supported:

- **T2VA**: text-to-video-and-audio generation;
- **FL2VA**: generation conditioned on a first frame, last frame, or both;
- **Ref2VA**: generation from ordered image, video, and audio references.

MiniMax-H3 weights are covered by the MiniMax H3 Community License. Review the
model card and license restrictions before downloading or running the model.

## Install

```sh
pip install -U mlx-vlm
```

`ffmpeg` and `ffprobe` must be available on `PATH` to extract embedded
soundtracks and write MP4 output. For example, on macOS:

```sh
brew install ffmpeg
```

The selected workflow downloads only its required transformer partition plus
the shared model components. You can also pass a local MiniMax-H3 snapshot or
converted MLX directory as the model path.

## Generic API and CLI

### CLI

Text-to-video-and-audio:

```sh
mlx_vlm.generate \
  --output-modality video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "A paper boat crossing a rain puddle" \
  --num-frames 124 \
  --steps 30 \
  --output outputs/paper-boat.mp4
```

First/last-frame conditioning:

```sh
mlx_vlm.generate \
  --output-modality video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "The scene changes from dawn to dusk" \
  --image first.png \
  --last-image last.png \
  --output outputs/fl2va.mp4
```

Ordered references:

```sh
mlx_vlm.generate \
  --output-modality video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "Keep the subject, motion, and soundtrack" \
  --reference image=subject.png \
  --reference video=motion.mp4 \
  --reference audio=soundtrack.wav \
  --output outputs/ref2va.mp4
```

The workflow is inferred from the conditioning arguments. It can also be set
explicitly with `--workflow t2va`, `--workflow fl2va`, or
`--workflow ref2va`. Reference order is significant, so repeat `--reference`
in the intended semantic order. A video reference automatically uses its
embedded soundtrack when present.

Use `--size WIDTHxHEIGHT` to select a canvas; both dimensions must be multiples
of 32. Video duration must be between 5 and 15 seconds, and frame counts are
aligned to the model's `17n+5` frame grid. When Ref2VA receives exactly one
audio-bearing reference, its duration can be inferred by omitting
`--num-frames`.

Generated files use H.264 video and AAC audio in an MP4 container. Without
`--output`, the CLI writes `outputs/video-<seed>.mp4`. Add `--verbose` for a
progress bar and effective frames-per-second reporting.

### Python

```python
from mlx_vlm.generate import (
    VideoGenerationRequest,
    generate_video,
    load_video_generation_model,
)

model = load_video_generation_model(
    "MiniMaxAI/MiniMax-H3",
    workflow="t2va",
)

result = generate_video(
    model,
    VideoGenerationRequest(
        prompt="A paper boat crossing a rain puddle",
        num_frames=124,
        steps=30,
        seed=42,
    ),
    output_path="outputs/paper-boat.mp4",
)

print(result.path, result.frames.shape, result.audio.shape)
```

For FL2VA, load with `workflow="fl2va"` and set `image` and/or `last_image` on
the request. For Ref2VA, load with `workflow="ref2va"` and pass ordered
`VideoReference` objects:

```python
from mlx_vlm.generate import VideoReference

request = VideoGenerationRequest(
    prompt="Keep the subject and soundtrack",
    references=(
        VideoReference("image", "subject.png"),
        VideoReference("audio", "soundtrack.wav"),
    ),
)
```

A loaded model is tied to one workflow. Reuse it with the same inference step
count, or reload it before changing workflows or step counts.
