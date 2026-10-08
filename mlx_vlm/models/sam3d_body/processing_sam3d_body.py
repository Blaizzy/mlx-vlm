"""Image preparation for SAM 3D Body."""

import json
from pathlib import Path

from ..base import install_auto_processor_patch
from .batch_prep import prepare_image


class SAM3DBodyProcessor:
    """Crops an image to a person box and normalizes it for the estimator."""

    def __init__(
        self,
        image_size=(512, 384),
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ):
        self.image_size = tuple(image_size)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config = Path(path) / "config.json"
        settings = json.loads(config.read_text()) if config.exists() else {}
        return cls(
            **{
                name: settings[name]
                for name in ("image_size", "image_mean", "image_std")
                if name in settings
            }
        )

    def __call__(self, image, bbox=None, **kwargs):
        if bbox is None:
            height, width = image.shape[:2]
            bbox = [0, 0, width, height]
        kwargs.setdefault("image_size", self.image_size)
        kwargs.setdefault("mean", self.image_mean)
        kwargs.setdefault("std", self.image_std)
        return prepare_image(image, bbox, **kwargs)


install_auto_processor_patch("sam3d_body", SAM3DBodyProcessor)
