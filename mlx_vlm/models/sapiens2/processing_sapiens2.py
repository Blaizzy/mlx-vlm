"""Sapiens2 image preprocessor.

Sapiens2 expects a fixed 1024 x 768 (H x W) RGB image normalized with ImageNet
statistics expressed on the [0, 255] scale (see
`sapiens/engine/datasets/data_preprocessors/image_preprocessor.py`):

  mean = [123.675, 116.28, 103.53]
  std  = [ 58.395,  57.12,  57.375]

The PyTorch reference treats the input as BGR (from OpenCV) and converts to RGB
as the first preprocessing step.  Our processor consumes PIL images (already RGB)
so the BGR→RGB flip is a no-op; keeping the same normalization constants makes
the feature distribution bit-identical to PT.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch


class Sapiens2Processor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 768),  # (H, W)
        image_mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        image_std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.image_size = tuple(image_size)
        # Keep on the [0, 255] scale to match PT's ImagePreprocessor exactly.
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json

        path = Path(path)
        image_size = (1024, 768)
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        cfg_path = path / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            if "image_size" in cfg:
                image_size = tuple(cfg["image_size"])
            if "image_mean" in cfg:
                mean = tuple(cfg["image_mean"])
            if "image_std" in cfg:
                std = tuple(cfg["image_std"])

        return cls(image_size=image_size, image_mean=mean, image_std=std)

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Returns {pixel_values: (1, H, W, 3), original_size: (orig_h, orig_w)}."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        H, W = self.image_size
        image = image.resize((W, H), Image.BILINEAR)
        pixel_values = np.array(image, dtype=np.float32)  # (H, W, 3) in [0, 255]
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        pixel_values = pixel_values[np.newaxis, ...]
        return {"pixel_values": pixel_values, "original_size": (orig_h, orig_w)}

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            items = [self.preprocess_image(im) for im in images]
            return {
                "pixel_values": np.concatenate([it["pixel_values"] for it in items], axis=0),
                "original_size": [it["original_size"] for it in items],
            }
        return self.preprocess_image(images)


install_auto_processor_patch(["sapiens2"], Sapiens2Processor)
