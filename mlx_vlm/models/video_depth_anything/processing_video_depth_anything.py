"""Video Depth Anything frame processor.

Mirrors the reference preprocessing: aspect-ratio-preserving resize to a
multiple of 14 (bicubic), then ImageNet normalization.
"""

from typing import Dict, Tuple

import mlx.core as mx
import numpy as np

from ..base import install_auto_processor_patch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoDepthProcessor:
    """Preprocess RGB frames (H, W, 3) uint8/float for the depth model."""

    def __init__(self, input_size: int = 518, ensure_multiple_of: int = 14):
        self.input_size = input_size
        self.ensure_multiple_of = ensure_multiple_of

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json
        from pathlib import Path

        cfg_path = Path(path) / "preprocessor_config.json"
        config = {}
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text())
        config.update(kwargs)
        known = ("input_size", "ensure_multiple_of")
        return cls(**{k: v for k, v in config.items() if k in known})

    def target_size(self, height: int, width: int) -> Tuple[int, int]:
        """Output (height, width), a multiple of 14, aspect ratio preserved."""
        input_size = self.input_size
        ratio = max(height, width) / min(height, width)
        if ratio > 1.78:  # long videos are limited to ~16:9 to save memory
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        scale = max(input_size / height, input_size / width)  # lower_bound
        new_h = max(round(height * scale / 14) * 14, input_size)
        new_w = max(round(width * scale / 14) * 14, input_size)
        return new_h, new_w

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize one RGB frame -> (H', W', 3) float32."""
        import cv2

        h, w = frame.shape[:2]
        new_h, new_w = self.target_size(h, w)
        frame = frame.astype(np.float32) / 255.0
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return (frame - IMAGENET_MEAN) / IMAGENET_STD

    def preprocess(self, frames: np.ndarray) -> Dict[str, mx.array]:
        """frames: (T, H, W, 3) uint8 RGB -> pixel_values (T, H', W', 3)."""
        out = [self.preprocess_frame(f) for f in frames]
        return {"pixel_values": mx.array(np.stack(out))}

    def __call__(self, frames: np.ndarray) -> Dict[str, mx.array]:
        return self.preprocess(frames)


install_auto_processor_patch(["video_depth_anything"], VideoDepthProcessor)
