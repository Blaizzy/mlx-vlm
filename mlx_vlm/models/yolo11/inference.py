import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw

from .yolo11 import YOLO11, load_weights, non_max_suppression

DEFAULT_MODEL_ID = "axiom-of-choice/OmniParser-v2-icon-detect"
ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class DetectionResult:
    boxes: mx.array
    scores: mx.array
    labels: mx.array
    image: Image.Image


def load_detector(model_path: Union[str, Path] = DEFAULT_MODEL_ID):
    """Load a converted OmniParser YOLO11 detector from disk or Hugging Face."""
    path = Path(model_path)
    if not path.exists():
        path = Path(snapshot_download(str(model_path)))
    if path.is_file():
        path = path.parent
    with open(path / "config.json") as f:
        config = json.load(f)
    model = YOLO11(
        nc=int(config["nc"]),
        ch=tuple(config.get("ch", (256, 512, 512))),
        reg_max=int(config.get("reg_max", 16)),
    )
    load_weights(model, mx.load(str(path / "model.safetensors")))
    return model


def prepare_image(image: ImageInput, size=None, stride=32):
    """Convert an image to a centered, stride-aligned detector input."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    width, height = image.size
    gain = 1.0 if size is None else min(float(size) / width, float(size) / height)
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    if (resized_width, resized_height) != (width, height):
        image_input = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
    else:
        image_input = image
    padded_width = math.ceil(resized_width / stride) * stride
    padded_height = math.ceil(resized_height / stride) * stride
    left = (padded_width - resized_width) // 2
    top = (padded_height - resized_height) // 2
    canvas = Image.new("RGB", (padded_width, padded_height), (114, 114, 114))
    canvas.paste(image_input, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return mx.array(array)[None], image, gain, left, top


def predict(
    model,
    image: ImageInput,
    size=None,
    conf_threshold=0.05,
    iou_threshold=0.1,
    max_detections=300,
):
    """Run detection and return boxes in original-image pixel coordinates."""
    inputs, original, gain, left, top = prepare_image(image, size=size)
    prediction = model(inputs)
    mx.eval(prediction)
    detections = non_max_suppression(
        prediction,
        conf_thresh=conf_threshold,
        iou_thresh=iou_threshold,
        max_det=max_detections,
    )[0]
    if detections.shape[0] == 0:
        return DetectionResult(
            boxes=mx.zeros((0, 4)),
            scores=mx.zeros((0,)),
            labels=mx.zeros((0,), dtype=mx.int32),
            image=original,
        )
    width, height = original.size
    boxes = detections[:, :4]
    x1 = mx.clip((boxes[:, 0] - left) / gain, 0, width)
    y1 = mx.clip((boxes[:, 1] - top) / gain, 0, height)
    x2 = mx.clip((boxes[:, 2] - left) / gain, 0, width)
    y2 = mx.clip((boxes[:, 3] - top) / gain, 0, height)
    return DetectionResult(
        boxes=mx.stack([x1, y1, x2, y2], axis=-1),
        scores=detections[:, 4],
        labels=detections[:, 5].astype(mx.int32),
        image=original,
    )


def draw_detections(result: DetectionResult):
    """Render numbered detection boxes on a copy of the source image."""
    image = result.image.copy()
    draw = ImageDraw.Draw(image)
    for index, (box, score) in enumerate(
        zip(result.boxes.tolist(), result.scores.tolist())
    ):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), f"{index} {score:.2f}", fill="red")
    return image
