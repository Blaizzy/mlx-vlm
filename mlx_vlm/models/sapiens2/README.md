# Sapiens2

MLX port of [Sapiens2](https://github.com/facebookresearch/sapiens2) (Meta,
ICLR 2026) — high-resolution vision transformers pretrained on 1B human
images for human-centric dense prediction. Unlike standard VLMs, these
models output dense predictions instead of text. Inference only.

The official Hugging Face repos load directly — `config.json`
(`model_type: "sapiens2"`) and the safetensors weights need no conversion.

## Models and tasks

| Task | Output | HF repos |
|------|--------|----------|
| Pose (308 keypoints) | heatmaps → keypoints | `mlx-community/sapiens2-pose-{0.4b,0.8b,1b,5b}-bf16` |
| Segmentation (29 parts) | per-pixel class logits | `mlx-community/sapiens2-seg-{0.4b,0.8b,1b,5b}-bf16` |
| Surface normals | per-pixel normals | `mlx-community/sapiens2-normal-{0.4b,0.8b,1b,5b}-bf16` |
| Pointmaps | per-pixel XYZ + scale | `mlx-community/sapiens2-pointmap-{0.4b,0.8b,1b,5b}-bf16` |
| Matting | alpha + foreground | `mlx-community/sapiens2-matting-1b-bf16` |

## Usage

```python
from mlx_vlm import load
from mlx_vlm.models.sapiens2.generate import Sapiens2Predictor, read_image

model, _ = load("mlx-community/sapiens2-seg-1b-bf16")
predictor = Sapiens2Predictor(model)
output = predictor.infer(read_image("image.jpg"))
```

`infer` takes an RGB image as an MLX or numpy array or a PIL image
(`read_image` returns uint8 MLX arrays) and returns a task-dependent dict
of MLX arrays; wrap a value in `np.array(...)` for a host copy:

- `backbone`: `last_hidden_state` (B, N, D), `pooler_output` (B, D)
- `seg`: `segmentation` (H, W) int32 class ids
- `pose`: `keypoints` (N, 308, 2), `scores` (N, 308), `boxes` — pass
  `boxes=` (xyxy) for cropped top-down inference; defaults to a full-image
  box. `flip_test=True` (default) also runs the mirrored crops and averages
  the heatmaps; pass `flip_test=False` for throughput.
- `normal`: `normals` (H, W, 3)
- `pointmap`: `pointmaps` (H, W, 3), `scales` (1,)
- `matting`: `alphas` (H, W), `foregrounds` (H, W, 3)

Dense outputs are resized back to the input resolution, and pose keypoints
are mapped back to source-image coordinates.

## Video and streaming

`infer` only builds the graph: nothing between the image and the outputs
waits on the GPU, and the returned arrays evaluate when read (`mx.eval`,
`np.array`, `.tolist()`). For a sequence of frames use `stream`, which
dispatches each frame with `mx.async_eval` and yields the previous frame's
outputs while the next one runs, so decoding, preprocessing and your
per-frame work overlap with the model:

```python
for out in predictor.stream("clip.mp4"):  # a path, a camera index, or any iterable of frames
    seg = np.array(out["segmentation"])  # blocks only until this frame is done
    frame = np.array(out["frame"])  # the RGB input frame, for overlays
```

`stream` takes a video path / camera index (decoded with OpenCV through
`read_video_frames`) or any iterable of images `infer` accepts, an optional
iterable of per-frame pose `boxes`, and `prefetch` (default 1), the number
of frames kept in flight.
