# OmniParser v2 icon detection

Native MLX implementation of the YOLO11 detector used by
`microsoft/OmniParser-v2.0`. The predictor accepts arbitrary image dimensions,
pads them to the model stride, applies non-maximum suppression, and returns
boxes in original-image pixel coordinates.

## Supported models

| Model | Component | Size |
|---|---|---:|
| [`axiom-of-choice/OmniParser-v2-icon-detect`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-detect) | YOLO11 icon detector | 80 MB |
| [`axiom-of-choice/OmniParser-v2-icon-caption-bf16`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-bf16) | Florence-2 icon captioner | 517 MB |
| [`axiom-of-choice/OmniParser-v2-icon-caption-8bit`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-8bit) | Florence-2 icon captioner | 356 MB |
| [`axiom-of-choice/OmniParser-v2-icon-caption-4bit`](https://huggingface.co/axiom-of-choice/OmniParser-v2-icon-caption-4bit) | Florence-2 icon captioner | 271 MB |

## Install

```sh
pip install -U mlx-vlm
```

## Python

```python
from mlx_vlm.models.yolo11 import draw_detections, load_detector, predict

model = load_detector("axiom-of-choice/OmniParser-v2-icon-detect")
result = predict(model, "screenshot.png", size=1280)

for box, score in zip(result.boxes.tolist(), result.scores.tolist()):
    print(f"{score:.3f} {box}")

draw_detections(result).save("screenshot-detections.png")
```

`size` sets the image's longest side before inference. Omit it to preserve the
native resolution. Both paths retain aspect ratio and pad each dimension to a
multiple of 32.

## Caption a detection

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

box = tuple(int(value) for value in result.boxes[0].tolist())
crop = result.image.crop(box).resize((64, 64))
crop.save("icon.png")

captioner, processor = load("axiom-of-choice/OmniParser-v2-icon-caption-8bit")
prompt = apply_chat_template(processor, captioner.config, "<CAPTION>", num_images=1)
output = generate(
    captioner,
    processor,
    prompt,
    image="icon.png",
    max_tokens=20,
    temperature=0.0,
)
print(output.text)
```

## Convert the original checkpoint

Conversion requires PyTorch and Ultralytics. Inference with the converted model
does not.

```sh
python -m mlx_vlm.models.yolo11.convert \
  --ckpt OmniParser-v2.0/icon_detect/model.pt \
  --output OmniParser-v2-icon-detect
```

The output directory contains `config.json` and `model.safetensors` and can be
passed directly to `load_detector`.

## Architecture

- YOLO11m backbone with C3k2 blocks
- SPPF and C2PSA neck
- Anchor-free detection head with DFL box decoding
- Depthwise-convolution classification branches
- Pure MLX class-aware NMS

## References

- [OmniParser](https://github.com/microsoft/OmniParser)
- [Original OmniParser v2 checkpoint](https://huggingface.co/microsoft/OmniParser-v2.0)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
