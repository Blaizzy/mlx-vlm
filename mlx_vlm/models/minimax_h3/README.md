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
used only at media or tokenizer I/O boundaries. `ffmpeg` is used to decode an
embedded soundtrack when a Ref2VA video path carries one. Runtime code does not
import PyTorch, Diffusers, torchvision, torchaudio, or librosa.

## License boundary

MiniMax-H3 weights use the MiniMax H3 Community License. Review the checkpoint
license and its territory/use restrictions before downloading, converting, or
running weights. The implementation and tests in this repository use only tiny
synthetic weights and do not download the checkpoint.

## Convert one partition

Conversion accepts a local official snapshot and never uploads artifacts. It
copies only one selected transformer, trims Qwen3-VL after decoder layer 50,
removes its final norm and LM head, and folds AudioVAE weight normalization.

```bash
python -m mlx_vlm.models.minimax_h3.convert \
  --source /path/to/MiniMax-H3 \
  --output /path/to/h3-fl2va-mlx \
  --partition fl2va
```

For a T2VA-only artifact, add `--text-only` to omit the Qwen vision tower. A
Ref2VA conversion always retains vision. Source dtypes are preserved unless
`--dtype` is supplied. Use `--dry-run` to validate the selected local files and
report their byte size without writing output.

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

Pass `image` and/or `last_image` for frame conditioning. Ref2VA uses a converted
`ref2va` directory and `MiniMaxH3Reference` objects in the request's
`references` list. Images may be paths, Pillow images, NumPy arrays, or MLX
arrays. Video/audio references may be decoded arrays or local container paths.
Generation is intentionally hardware-neutral: no automatic offload, streaming,
quantization recipe, or minimum-memory claim is imposed.

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
