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

## CLI

```bash
python -m mlx_vlm generate \
    --model tiiuae/Falcon-Perception \
    --image photo.jpg \
    --prompt "cat" \
    --max-tokens 200
```

## Python

### Detection + Segmentation

```python
from mlx_vlm import load, generate

model, processor = load("tiiuae/Falcon-Perception")

output = generate(
    model,
    processor,
    "cat",
    image="photo.jpg",
    max_tokens=200,
    verbose=True,
)

# Retrieve detections accumulated during generate()
detections = model.get_detections()

for det in detections:
    xy, hw = det["xy"], det["hw"]
    has_mask = "mask" in det
    print(f"Center: ({xy['x']:.3f}, {xy['y']:.3f}), Size: ({hw['h']:.3f}, {hw['w']:.3f}), Mask: {has_mask}")
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
- `mask` -- `(H, W)` binary `mx.array` segmentation mask (present when `<|seg|>` token is decoded)

### Notes

- Segmentation masks are produced at patch resolution and upsampled via nearest-neighbor interpolation. The original PyTorch model uses AnyUp (a learned cross-attention upsampler with FlexAttention) for higher-resolution masks, which is not available in MLX.
- The coord/size Fourier encoding feedback is handled automatically by the `LanguageModel` during standard `generate()`. Use `model.get_detections()` after generation to retrieve results.
