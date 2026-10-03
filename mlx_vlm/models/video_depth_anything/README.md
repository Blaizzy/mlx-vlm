# Video Depth Anything

MLX port of [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) (ByteDance, CVPR 2025 highlight): consistent monocular depth estimation for arbitrarily long videos.

Architecture: DINOv2 backbone (vits/vitb/vitl) + DPT head with AnimateDiff-style temporal motion modules. The model outputs per-frame depth maps, not text.

## Supported checkpoints

| Variant | MLX repo | Source repo |
|:-|:-|:-|
| Small | `mlx-community/Video-Depth-Anything-Small-MLX` | `depth-anything/Video-Depth-Anything-Small` |
| Base | `mlx-community/Video-Depth-Anything-Base-MLX` | `depth-anything/Video-Depth-Anything-Base` |
| Large | `mlx-community/Video-Depth-Anything-Large-MLX` | `depth-anything/Video-Depth-Anything-Large` |
| Small metric | `mlx-community/Metric-Video-Depth-Anything-Small-MLX` | `depth-anything/Metric-Video-Depth-Anything-Small` |
| Base metric | `mlx-community/Metric-Video-Depth-Anything-Base-MLX` | `depth-anything/Metric-Video-Depth-Anything-Base` |
| Large metric | `mlx-community/Metric-Video-Depth-Anything-Large-MLX` | `depth-anything/Metric-Video-Depth-Anything-Large` |

The loader downloads the weights from the Hub; no conversion step is needed.

## Usage

```python
from mlx_vlm import load
from mlx_vlm.models.video_depth_anything.generate import (
    VideoDepthPredictor,
    read_video_frames,
)

model, processor = load("mlx-community/Video-Depth-Anything-Small-MLX")
predictor = VideoDepthPredictor(model, processor)

frames, fps = read_video_frames("input.mp4", max_len=300, target_fps=15)
depths = predictor.infer(frames)  # (T, H, W) float32, input resolution
```

`VideoDepthPredictor.infer` runs the reference sliding-window inference: overlapping 32-frame windows, keyframe conditioning, and scale/shift alignment between windows (skipped for metric models). For a single short clip you can call the model directly:

```python
import numpy as np
frames = ...  # (T, H, W, 3) uint8 RGB
pixel_values = processor(frames)["pixel_values"]  # (T, H', W', 3) normalized
depth = model.predict_depth(pixel_values)  # (T, H', W')
```

## Notes

- Inputs are channel-last `(B, T, H, W, 3)`; H and W must be multiples of 14.
- On the default GPU device the output matches the PyTorch reference to ~1%
  relative (Metal fast-math matmuls). On CPU (`mx.set_default_device(mx.cpu)`)
  it matches to ~1e-5 relative, including the sliding-window pipeline.
- The streaming mode of the reference (`video_depth_stream.py`) is not ported.
