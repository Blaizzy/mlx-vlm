"""RT-DETRv2 image preprocessing.

The standard RT-DETRv2 preprocessor: resize to 640x640 bilinear, rescale
by 1/255, no mean/std normalization. The "no normalization" detail is
unusual for ImageNet-style vision models, and silently adding mean/std
subtraction is the most common way to get subtly-wrong outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

ImageInput = Union[Image.Image, np.ndarray]


@dataclass
class RTDetrV2ProcessorConfig:
    image_size: int = 640
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = False


@dataclass
class ProcessorOutput:
    pixel_values: mx.array  # (B, image_size, image_size, 3) NHWC in [0, 1]
    original_sizes: List[Tuple[int, int]]  # per-image (width, height) for box rescaling


class RTDetrV2Processor:
    """Batched preprocessor for RT-DETRv2."""

    def __init__(self, config: Optional[RTDetrV2ProcessorConfig] = None) -> None:
        self.config = config or RTDetrV2ProcessorConfig()

    def __call__(
        self,
        images: Union[ImageInput, Iterable[ImageInput]],
    ) -> ProcessorOutput:
        batch = (
            [images] if isinstance(images, (Image.Image, np.ndarray)) else list(images)
        )
        if not batch:
            raise ValueError("Empty image batch")

        size = self.config.image_size
        rescale = self.config.rescale_factor

        original_sizes: List[Tuple[int, int]] = []
        arrays: List[np.ndarray] = []
        for img in batch:
            pil = _to_pil_rgb(img)
            original_sizes.append(pil.size)
            resized = pil.resize((size, size), Image.Resampling.BILINEAR)
            arrays.append(np.asarray(resized, dtype=np.float32))

        stacked = np.stack(arrays, axis=0) * rescale
        return ProcessorOutput(
            pixel_values=mx.array(stacked),
            original_sizes=original_sizes,
        )


def _to_pil_rgb(img: ImageInput) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img).__name__}")
