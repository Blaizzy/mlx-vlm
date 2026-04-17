# Falcon-Perception

Falcon-Perception is an early-fusion vision-language model family from TII for object detection and segmentation. It generates bounding box coordinates, sizes, and segmentation masks for objects matching a text query, using a single Transformer that processes image patches and text tokens in a shared parameter space.

## Models

| Model ID | Parameters | Detection | Segmentation |
|----------|-----------|-----------|-------------|
| `tiiuae/Falcon-Perception` | ~0.6B | Yes | Yes |
| `tiiuae/Falcon-Perception-300M` | ~0.3B | Yes | No |

### Links

- [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) -- code and inference engine
- [tiiuae/Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) -- HuggingFace model card

## Installation

```bash
pip install mlx-vlm
```

## Python

### Detection + Segmentation

```python
from mlx_vlm import load

model, processor = load("tiiuae/Falcon-Perception")

detections = model.generate_perception(
    processor,
    image="photo.jpg",
    query="cat",
    max_new_tokens=512,
)

for det in detections:
    xy, hw = det["xy"], det["hw"]
    has_mask = "mask" in det
    print(f"Center: ({xy['x']:.3f}, {xy['y']:.3f}), Size: ({hw['h']:.3f}, {hw['w']:.3f}), Mask: {has_mask}")
```

### Detection Only (300M)

```python
from mlx_vlm import load

model, processor = load("tiiuae/Falcon-Perception-300M")

detections = model.generate_perception(
    processor,
    image="photo.jpg",
    query="cats",
    max_new_tokens=512,
)
```

### Plotting Detections

```python
from mlx_vlm.models.falcon_perception import plot_detections

plot_detections("photo.jpg", detections, save_path="output.png")
```

See [`examples/falcon_perception_demo.ipynb`](../../../examples/falcon_perception_demo.ipynb) for a complete example with visualization.

### Output Format

Each detection is a dict with:
- `xy` -- center coordinates `{"x": float, "y": float}` normalized to `[0, 1]`
- `hw` -- bounding box size `{"h": float, "w": float}` as fraction of image dimensions
- `mask` -- `(H, W)` binary `mx.array` segmentation mask (only on the 0.6B model)
