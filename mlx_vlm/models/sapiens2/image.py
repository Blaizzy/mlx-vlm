"""Sapiens2 image preprocessing.

Images are RGB ``(H, W, 3)`` or ``(B, H, W, 3)`` arrays in [0, 255] (MLX or
numpy arrays, or PIL images); outputs are ImageNet-normalized float32 MLX
arrays. Sizes are ``(height, width)``.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from PIL import Image

from ..base import install_auto_processor_patch
from ..interpolate import resize_bicubic_nhwc, resize_bilinear_nhwc

IMAGENET_MEAN = mx.array([123.675, 116.28, 103.53])
IMAGENET_STD = mx.array([58.395, 57.12, 57.375])

# Tasks whose test pipeline pads instead of stretching the image.
PAD_TASKS = {"normal", "pointmap"}

# An MLX array, a numpy array or a PIL image.
ImageLike = Any


def _from_pil(image: Image.Image) -> mx.array:
    """PIL image -> uint8 (H, W, 3) array, without a numpy round trip."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return mx.array(image.tobytes()).reshape(image.height, image.width, 3)


def to_array(image: ImageLike) -> mx.array:
    """MLX/numpy/PIL image -> MLX array, copied to device only when needed."""
    if isinstance(image, mx.array):
        return image
    if isinstance(image, Image.Image):
        return _from_pil(image)
    if any(s < 0 for s in getattr(image, "strides", ())):
        image = image.copy()  # MLX cannot import negative-stride (flipped) views
    return mx.array(image)


def to_batch(image: ImageLike) -> mx.array:
    """(H, W, C) or (B, H, W, C) image -> (B, H, W, C) float32."""
    x = to_array(image).astype(mx.float32)
    return x if x.ndim == 4 else x[None]


def resize_image(img: ImageLike, size: Tuple[int, int]) -> mx.array:
    """Resize to ``size`` (h, w); antialiased bilinear downscale, bicubic upscale."""
    x = to_batch(img)
    h, w = x.shape[1:3]
    if (h, w) == tuple(size):
        return x
    if size[1] < w:
        return resize_bilinear_nhwc(x, size, antialias=True)
    return resize_bicubic_nhwc(x, size)


def resize_pad_image(
    img: ImageLike, size: Tuple[int, int], pad_val: int = 0
) -> Tuple[mx.array, Tuple[int, int, int, int]]:
    """Aspect-ratio bilinear resize to fit ``size`` (h, w), then symmetric padding.

    Returns the padded batch and ``(left, right, top, bottom)`` padding.
    """
    x = to_batch(img)
    target_h, target_w = size
    h, w = x.shape[1:3]
    scale = min(target_w / w, target_h / h)
    new_h, new_w = int(h * scale), int(w * scale)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    bottom, right = target_h - new_h - top, target_w - new_w - left
    resized = resize_bilinear_nhwc(x, (new_h, new_w))
    padded = mx.pad(
        resized, [(0, 0), (top, bottom), (left, right), (0, 0)], constant_values=pad_val
    )
    return padded, (left, right, top, bottom)


def unpad_image(img, padding: Tuple[int, int, int, int]):
    """Remove ``(left, right, top, bottom)`` padding from an (H, W, ...) image/map."""
    left, right, top, bottom = padding
    h, w = img.shape[:2]
    return img[top : h - bottom, left : w - right]


def normalize_image(img: ImageLike) -> mx.array:
    """ImageNet-normalize an RGB image or batch in [0, 255] -> float32."""
    return (to_array(img) - IMAGENET_MEAN) / IMAGENET_STD


def preprocess(
    image: ImageLike, task: str, size: Tuple[int, int] = (1024, 768)
) -> Dict:
    """Preprocess an RGB image for the given task.

    Returns ``pixel_values`` (1, h, w, 3) float32, the ``padding`` (padded
    tasks only) and the ``input_size`` (h, w) fed to the model.
    """
    if task in PAD_TASKS:
        img, padding = resize_pad_image(image, size)
    else:
        img, padding = resize_image(image, size), None
    return {
        "pixel_values": normalize_image(img),
        "padding": padding,
        "input_size": size,
    }


class Sapiens2Processor:
    """Image processor matching the official ``preprocessor_config.json``:
    resize (optionally with padding), rescale, ImageNet normalization."""

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        do_pad: bool = False,
        do_normalize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        size = size or {"height": 1024, "width": 768}
        self.size = (size["height"], size["width"])  # (H, W)
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.image_mean = mx.array(image_mean) if image_mean else IMAGENET_MEAN / 255
        self.image_std = mx.array(image_std) if image_std else IMAGENET_STD / 255
        scale = rescale_factor if do_rescale else 1.0
        self.scale = scale / self.image_std if do_normalize else scale
        self.offset = -self.image_mean / self.image_std if do_normalize else 0.0

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json
        from pathlib import Path

        cfg_path = Path(path) / "preprocessor_config.json"
        config = {}
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text())
        config.update(kwargs)
        return cls(**config)

    def preprocess_image(self, image: ImageLike) -> mx.array:
        """RGB (H, W, 3) or (B, H, W, 3) in [0, 255] -> (B, H, W, 3) float32."""
        if self.do_pad:
            image, _ = resize_pad_image(image, self.size)
        else:
            image = resize_image(image, self.size)
        return image * self.scale + self.offset

    def preprocess(self, images: Union[ImageLike, List[ImageLike]]) -> Dict:
        if isinstance(images, (list, tuple)):
            pixel_values = mx.concatenate([self.preprocess_image(i) for i in images])
        else:
            pixel_values = self.preprocess_image(images)
        return {"pixel_values": pixel_values}

    def __call__(self, images, **kwargs) -> Dict:
        return self.preprocess(images)


install_auto_processor_patch(["sapiens2"], Sapiens2Processor)
