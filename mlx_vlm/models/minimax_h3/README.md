# MiniMax-H3

This directory contains the MLX-native MiniMax-H3 Base implementation for joint
24 FPS video and 32 kHz stereo-audio generation. It supports:

- T2VA and first/last-frame conditioning with the `fl2va` transformer;
- ordered image, video, audio, and video-with-soundtrack references with the
  `ref2va` transformer;
- strict loading from an official local checkpoint or a converted MLX
  directory;
- MLX-native resize, normalization, patchification, frame resampling, waveform
  resampling, sequence packing, transformer, VisualVAE, and AudioVAE math.

Pillow, OpenCV, `mlx-audio`, `ffprobe`/`ffmpeg`, and the tokenizer backend are
used only at media or tokenizer I/O boundaries. `ffmpeg` decodes an embedded
soundtrack when a Ref2VA video path carries one and encodes/muxes generated RGB
frames and stereo audio into MP4 output. Runtime code does not import PyTorch,
Diffusers, torchvision, torchaudio, or librosa.

## License boundary

MiniMax-H3 weights use the MiniMax H3 Community License. Review the checkpoint
license and its territory/use restrictions before downloading, converting, or
running weights. The implementation and tests in this repository use only tiny
synthetic weights and do not download the checkpoint.

## Convert one partition

Conversion accepts a local official snapshot and never uploads artifacts. It
copies only one selected transformer, trims Qwen3-VL after decoder layer 50,
removes its final norm and LM head, and folds AudioVAE weight normalization.

Download the public Diffusers layout by selecting one workflow first. This
fetches the shared components once and only `transformer/` for `t2va`/`fl2va`
or `transformer_ref/` for `ref2va`; it never enters the duplicated legacy
`FL2VA/` and `Ref2VA/` trees.

```python
from mlx_vlm.models.minimax_h3 import download_model

source = download_model(workflow="fl2va")
print(source)
```

```bash
python -m mlx_vlm.models.minimax_h3.convert \
  --source MiniMaxAI/MiniMax-H3 \
  --output /path/to/h3-fl2va-mlx \
  --workflow fl2va
```

`--source` also accepts an already downloaded local snapshot. For a T2VA-only
artifact, add `--text-only` to omit the Qwen vision tower. A Ref2VA conversion
always retains vision. Source dtypes are preserved unless `--dtype` is supplied.
Use `--dry-run` to validate the selected files and report their byte size without
writing output.

## Load and generate

```python
from mlx_vlm.models.minimax_h3 import (
    MiniMaxH3GenerationRequest,
    load_pipeline,
)

pipeline = load_pipeline("/path/to/h3-fl2va-mlx")
result = pipeline.generate(
    MiniMaxH3GenerationRequest(
        prompt="A paper boat crossing a rain puddle",
        num_frames=124,
        num_inference_steps=30,
    )
)
```

An official Hub repository can be loaded directly. Remote loading requires a
Diffusers-style workflow so it cannot accidentally fetch both 61.7 GB
transformers:

```python
pipeline = load_pipeline(
    "MiniMaxAI/MiniMax-H3",
    workflow="t2va",  # or "fl2va" / "ref2va"
)
```

Pass `image` and/or `last_image` for frame conditioning. Ref2VA uses a converted
`ref2va` directory and `MiniMaxH3Reference` objects in the request's
`references` list. Images may be paths, Pillow images, NumPy arrays, or MLX
arrays. Video/audio references may be decoded arrays or local container paths.
Generation is intentionally hardware-neutral: no automatic offload, streaming,
quantization recipe, or minimum-memory claim is imposed.

Before denoising, the pipeline now materializes each block's AdaLN modulation
using the exact per-step timestep tensor that the uncached forward would use.
The cache preserves its shape, ordering, and dtype, and is fully evaluated
before the roughly 13B per-block AdaLN projection parameters are released. The
same loaded pipeline can be reused with the same schedule and conditioning
mode. After the projection weights have been released, changing the number of
steps or switching to a mode with a different timestep table requires reloading
the pipeline. Python callers that need to retain a mutable/reusable transformer
can set `drop_adaln_weights=False`; set `cache_adaln=False` as well to exercise
the original live path.

### Generic API and CLI

Video generation is also available through the public generation API. The
workflow is selected when the model is loaded so only its required transformer
partition is downloaded:

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
    ),
    output_path="outputs/paper-boat.mp4",
)
```

The regular generator accepts `--output-modality video`; `generate_video` is a
convenience subcommand for the same path. The mode is inferred from the supplied
conditioning, or it can be selected explicitly with `--workflow`:

```bash
# T2VA: no frame or reference conditioning
mlx_vlm.generate \
  --output-modality video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "A paper boat crossing a rain puddle" \
  --num-frames 124 \
  --verbose \
  --output outputs/t2va.mp4

# FL2VA: first frame, last frame, or both
python -m mlx_vlm generate_video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "The scene changes from dawn to dusk" \
  --image first.png \
  --last-image last.png \
  --output outputs/fl2va.mp4

# Ref2VA: repeat --reference to preserve semantic order
mlx_vlm.generate \
  --output-modality video \
  --model MiniMaxAI/MiniMax-H3 \
  --prompt "Keep the subject, motion, and soundtrack" \
  --reference image=subject.png \
  --reference video=motion.mp4 \
  --reference audio=soundtrack.wav \
  --output outputs/ref2va.mp4
```

Generated files use H.264 video and AAC audio in an MP4 container. `ffmpeg` must
be installed and visible on `PATH`. Without `--output`, the CLI writes
`outputs/video-<seed>.mp4`.

Pass `--verbose` to display a phase-aware tqdm progress bar. It reports model
loading, conditioning preparation, AdaLN cache construction, every denoising
update, audio/video decode, and MP4 encoding. During generation, `frames/s` is
the effective output-frame throughput through the completed fraction of
denoising; the final summary reports end-to-end `generation_fps`. The progress
bar is off by default. H3 denoises the complete clip jointly, so frames are not
emitted one at a time.

## Synthetic parity

The focused tests cover exact layouts and scheduler behavior plus numeric
goldens produced from the pinned references:

- Diffusers MiniMax-H3 transformer;
- Diffusers VisualVAE encode/decode, temporal chunking, and spatial tiling;
- Diffusers AudioVAE encode/decode;
- Transformers Qwen3-VL pre-final-norm decoder hidden states;
- Transformers/Pillow processor behavior;
- full tiny FL2VA and Ref2VA denoising requests;
- official-key-layout conversion and strict reload for both partitions.

Reference revisions are recorded in `PLAN.md` and every converted
`h3_manifest.json`.
