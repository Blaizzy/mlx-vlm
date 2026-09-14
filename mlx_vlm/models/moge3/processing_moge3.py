"""MoGe-3 image processor.

MoGe works on the native image resolution, so preprocessing is only an
optional downscale plus a [0, 1] rescale. ImageNet normalization happens
inside the model.
"""

from typing import Dict, Optional

import mlx.core as mx
import numpy as np

from ..base import install_auto_processor_patch


class MogeProcessor:
    """Preprocess RGB images (H, W, 3) or (B, H, W, 3) for MoGe-3."""

    def __init__(self, resize_to: Optional[int] = None):
        # Optional longest-side cap; downscaled with INTER_AREA as in the
        # reference CLI.
        self.resize_to = resize_to

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json
        from pathlib import Path

        cfg_path = Path(path) / "preprocessor_config.json"
        config = {}
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text())
        config.update(kwargs)
        return cls(**{k: v for k, v in config.items() if k in ("resize_to",)})

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) or (B, H, W, 3) -> float32 in [0, 1] (optionally downscaled).

        Integer images are scaled by their dtype range (uint8 -> /255,
        uint16 -> /65535). Float images must already be in [0, 1].
        """
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32) / np.iinfo(image.dtype).max
        else:
            image = image.astype(np.float32)
            if image.size and (image.min() < 0.0 or image.max() > 1.0):
                raise ValueError(
                    "Float images must be in [0, 1]; pass integer images "
                    "(uint8/uint16) or scale the array first."
                )
        if self.resize_to is not None:
            image = self._downscale(image)
        return image

    def _downscale(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[-3:-1]
        limit = self.resize_to
        height, width = (
            min(limit, int(limit * height / width)),
            min(limit, int(limit * width / height)),
        )
        if image.ndim == 3:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.stack(
            [
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                for frame in image
            ]
        )

    def preprocess(self, image: np.ndarray) -> Dict[str, mx.array]:
        return {"pixel_values": mx.array(self.preprocess_image(image))}

    def __call__(self, image: np.ndarray) -> Dict[str, mx.array]:
        return self.preprocess(image)


install_auto_processor_patch(["moge3"], MogeProcessor)
