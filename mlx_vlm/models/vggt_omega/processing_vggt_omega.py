"""VGGT-Omega image preprocessing, as lazy MLX ops.

Mirrors the reference ``load_and_preprocess_images``: transparent pixels
become white, extreme aspect ratios are center-cropped into [0.5, 2.0],
each image is resized (bicubic) to a patch-multiple size, and mixed sizes
are padded with white to a common size.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import mlx.core as mx
from PIL import Image, ImageOps

from ..base import install_auto_processor_patch
from ..interpolate import resize_bicubic_nhwc
from ..sapiens2.image import to_array

# A path, URL or data URI, a PIL image, or an (H, W, 3) RGB array (uint8,
# or float in [0, 1]).
ImageLike = Any


def _flatten_alpha(image: Image.Image) -> Image.Image:
    if not image.has_transparency_data:
        return image
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image)


def read_image(source: ImageLike) -> mx.array:
    """Read an image as RGB (H, W, 3); EXIF orientation is applied and
    transparent pixels become white. Arrays pass through."""
    from ...utils import load_image

    if isinstance(source, (str, Path)):
        if str(source).startswith(("http://", "https://", "data:")):
            return to_array(load_image(source))
        with Image.open(source) as image:
            return read_image(image)
    if isinstance(source, Image.Image):
        return to_array(_flatten_alpha(ImageOps.exif_transpose(source)))
    return to_array(source)


def crop_to_aspect_ratio(
    image: mx.array, min_ratio: float = 0.5, max_ratio: float = 2.0
) -> mx.array:
    """Center-crop (H, W, C) so that H / W is in [min_ratio, max_ratio]."""
    height, width = image.shape[:2]
    ratio = height / max(width, 1)
    if ratio < min_ratio:
        crop = min(width, max(1, round(height / min_ratio)))
        left = (width - crop) // 2
        return image[:, left : left + crop]
    if ratio > max_ratio:
        crop = min(height, max(1, round(width * max_ratio)))
        top = (height - crop) // 2
        return image[top : top + crop]
    return image


def _round_half_up(x: mx.array) -> mx.array:
    return mx.clip(mx.floor(x + 0.5), 0, 255)


def resize_like_pil(image: mx.array, size: Tuple[int, int]) -> mx.array:
    """PIL ``BICUBIC`` resize of an (H, W, 3) image to ``size`` (h, w), as
    float32 in [0, 255].

    uint8 images follow PIL's two passes (width, then height), each rounded
    to uint8 levels; float images are resized in one pass without rounding.
    """
    x = image.astype(mx.float32)[None]
    if image.dtype != mx.uint8:
        return resize_bicubic_nhwc(x * 255, size, antialias=True)[0]
    height = x.shape[1]
    x = _round_half_up(resize_bicubic_nhwc(x, (height, size[1]), antialias=True))
    x = _round_half_up(resize_bicubic_nhwc(x, size, antialias=True))
    return x[0]


class VGGTOmegaProcessor:
    """Turns a sequence of images into ``pixel_values`` (S, H, W, 3) in [0, 1].

    ``mode="balanced"`` keeps the patch count near
    ``(image_resolution / patch_size) ** 2``; ``"max_size"`` resizes the
    longest side to ``image_resolution``.
    """

    def __init__(
        self,
        image_resolution: int = 512,
        mode: str = "balanced",
        patch_size: int = 16,
        **kwargs,
    ):
        if mode not in ("balanced", "max_size"):
            raise ValueError("mode must be 'balanced' or 'max_size'")
        if image_resolution <= 0 or image_resolution % patch_size != 0:
            raise ValueError(
                "image_resolution must be a positive multiple of patch_size"
            )
        self.image_resolution = image_resolution
        self.mode = mode
        self.patch_size = patch_size

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_path = Path(path) / "preprocessor_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        config.update(kwargs)
        return cls(**config)

    def target_size(self, height: int, width: int) -> Tuple[int, int]:
        """Model input size (h, w) for an image of this (cropped) size."""
        p, res = self.patch_size, self.image_resolution
        ratio = height / max(width, 1)
        if self.mode == "balanced":
            tokens = (res // p) ** 2
            w_patches = math.sqrt(tokens / ratio)
            h_patches = tokens / w_patches
            return max(1, round(h_patches)) * p, max(1, round(w_patches)) * p

        def to_patches(value):
            return max(p, round(value / p) * p)

        if ratio >= 1.0:
            return res, to_patches(res / ratio)
        return to_patches(res * ratio), res

    def preprocess_image(self, image: ImageLike) -> mx.array:
        """One image -> (h, w, 3) float32 in [0, 1]."""
        image = crop_to_aspect_ratio(read_image(image))
        size = self.target_size(*image.shape[:2])
        return resize_like_pil(image, size) / 255.0

    def preprocess(self, images: Sequence[ImageLike]) -> Dict[str, mx.array]:
        if isinstance(images, mx.array) and images.ndim == 3:
            images = [images]
        frames = [self.preprocess_image(image) for image in images]
        if not frames:
            raise ValueError("At least 1 image is required")
        shapes = {frame.shape[:2] for frame in frames}
        if len(shapes) > 1:
            warnings.warn(
                f"Found images with different shapes: {shapes}; padding to a "
                "common size.",
                stacklevel=2,
            )
            height = max(h for h, _ in shapes)
            width = max(w for _, w in shapes)
            frames = [_pad_white(frame, height, width) for frame in frames]
        return {"pixel_values": mx.stack(frames)}

    def __call__(self, images: Sequence[ImageLike], **kwargs) -> Dict[str, mx.array]:
        return self.preprocess(images)


def _pad_white(frame: mx.array, height: int, width: int) -> mx.array:
    dh, dw = height - frame.shape[0], width - frame.shape[1]
    pad = [(dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2), (0, 0)]
    return mx.pad(frame, pad, constant_values=1.0)


install_auto_processor_patch(["vggt_omega"], VGGTOmegaProcessor)
