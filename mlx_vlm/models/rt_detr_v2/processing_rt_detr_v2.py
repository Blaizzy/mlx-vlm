"""RT-DETRv2 image preprocessing.

The standard RT-DETRv2 preprocessor: resize to `image_size` bilinear,
rescale by `rescale_factor` (1/255), no mean/std normalization. The
"no normalization" detail is unusual for ImageNet-style vision models,
and silently adding mean/std subtraction is the most common way to get
subtly-wrong outputs.

`from_pretrained` reads `preprocessor_config.json` if present so the
runtime config tracks the saved checkpoint, and the
`install_auto_processor_patch` call at the bottom makes
`transformers.AutoProcessor.from_pretrained` dispatch RT-DETRv2
directories here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch

ImageInput = Union[Image.Image, np.ndarray]


@dataclass
class RTDetrV2ProcessorConfig:
    image_size: int = 640
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = False
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ProcessorOutput:
    pixel_values: mx.array  # (B, image_size, image_size, 3) NHWC in [0, 1]
    original_sizes: List[Tuple[int, int]]  # per-image (width, height) for box rescaling


class RTDetrV2Processor:
    """Batched preprocessor for RT-DETRv2."""

    def __init__(self, config: Optional[RTDetrV2ProcessorConfig] = None) -> None:
        self.config = config or RTDetrV2ProcessorConfig()

    @classmethod
    def from_pretrained(cls, path, **kwargs) -> "RTDetrV2Processor":
        """Build a processor from a model directory.

        Reads `preprocessor_config.json` for resize/rescale/normalize fields
        and falls back to `config.json["image_size"]` for the resize target
        if the preprocessor file is missing. Matches the HF
        `RTDetrImageProcessor` schema (`size: {height, width}`,
        `rescale_factor`, `do_normalize`, `image_mean`, `image_std`).
        """
        model_dir = Path(path)
        cfg = RTDetrV2ProcessorConfig()

        preproc_file = model_dir / "preprocessor_config.json"
        if preproc_file.is_file():
            pp = json.loads(preproc_file.read_text())
            size = pp.get("size")
            if isinstance(size, dict):
                # HF stores {"height": H, "width": W}; we require square inputs.
                cfg.image_size = int(
                    size.get("height", size.get("shortest_edge", cfg.image_size))
                )
            elif isinstance(size, int):
                cfg.image_size = size
            cfg.rescale_factor = float(pp.get("rescale_factor", cfg.rescale_factor))
            cfg.do_normalize = bool(pp.get("do_normalize", cfg.do_normalize))
            mean = pp.get("image_mean")
            std = pp.get("image_std")
            if mean is not None:
                cfg.image_mean = tuple(float(x) for x in mean)
            if std is not None:
                cfg.image_std = tuple(float(x) for x in std)
        else:
            config_file = model_dir / "config.json"
            if config_file.is_file():
                model_cfg = json.loads(config_file.read_text())
                cfg.image_size = int(model_cfg.get("image_size", cfg.image_size))

        return cls(cfg)

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
        if self.config.do_normalize:
            mean = np.asarray(self.config.image_mean, dtype=np.float32)
            std = np.asarray(self.config.image_std, dtype=np.float32)
            stacked = (stacked - mean) / std
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


# Register with transformers.AutoProcessor so `AutoProcessor.from_pretrained`
# on a directory whose config.json has `model_type: rt_detr_v2` dispatches
# here. Defined in mlx_vlm/models/base.py:417.
install_auto_processor_patch(["rt_detr_v2"], RTDetrV2Processor)
