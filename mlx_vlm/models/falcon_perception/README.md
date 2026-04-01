# Falcon-Perception

Falcon-Perception is a 1B parameter early-fusion vision-language model from TII for object detection and segmentation. It generates bounding box coordinates, sizes, and segmentation masks for objects matching a text query, using a single Transformer that processes image patches and text tokens in a shared parameter space.

## Model

- **Model ID**: `tiiuae/Falcon-Perception`
- **Parameters**: ~1B
- **Tasks**: Object detection (bounding boxes), referring expression segmentation (masks)

### Links

- [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) -- code and inference engine
- [tiiuae/Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) -- HuggingFace model card

## Installation

```bash
pip install mlx-vlm
```

## Python Examples

### Detection + Segmentation

Use `generate_perception` for the full detection pipeline with coord/size/mask decoding:

```python
from mlx_vlm import load
from mlx_vlm.models.falcon_perception import generate_perception, plot_detections

model, processor = load("tiiuae/Falcon-Perception")

detections = generate_perception(
    model, processor,
    image="photo.jpg",
    query="cat",
    max_new_tokens=200,
)

for det in detections:
    xy, hw = det["xy"], det["hw"]
    has_mask = "mask" in det
    print(f"Center: ({xy['x']:.3f}, {xy['y']:.3f}), Size: ({hw['h']:.3f}, {hw['w']:.3f}), Mask: {has_mask}")
```

### Plotting Detections

```python
plot_detections("photo.jpg", detections, save_path="output.png")
```

### Output Format

Each detection is a dict with:
- `xy` -- center coordinates `{"x": float, "y": float}` normalized to `[0, 1]`
- `hw` -- bounding box size `{"h": float, "w": float}` as fraction of image dimensions
- `mask` -- `(H, W)` binary `mx.array` segmentation mask (present when `<|seg|>` token is decoded)

### Notes

- Segmentation masks are produced at patch resolution and upsampled via nearest-neighbor interpolation. The original PyTorch model uses AnyUp (a learned cross-attention upsampler with FlexAttention) for higher-resolution masks, which is not available in MLX.
- The `generate_perception` function implements a custom decode loop with coord/size Fourier encoding feedback, which is required for accurate detection. Standard `generate()` will produce detection tokens but without decoded coordinates.
