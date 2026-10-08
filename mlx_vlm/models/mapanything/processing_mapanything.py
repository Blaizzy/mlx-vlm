"""MapAnything input preprocessing (ports of ``mapanything.utils.image``
``load_images`` and ``preprocess_inputs``).

All views are resized to one size picked from their mean aspect ratio: Pillow
LANCZOS when shrinking, BICUBIC when enlarging (two uint8-rounded passes, as
Pillow does), then center-cropped. Depth maps follow with OpenCV
``INTER_NEAREST`` and intrinsics are rescaled and shifted to match. Images are
decoded on the host; everything after that is lazy MLX work on the GPU.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch
from ..interpolate import resize_bicubic_nhwc, resize_lanczos_nhwc
from .geometry import intrinsics_from_rays
from .mapanything import IMAGE_NORMALIZATIONS, _as_array

# (width, height) per aspect ratio (width / height) for each resolution set.
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),
        1.321: (518, 392),
        1.542: (518, 336),
        1.762: (518, 294),
        2.056: (518, 252),
        3.083: (518, 168),
        0.757: (392, 518),
        0.649: (336, 518),
        0.567: (294, 518),
        0.486: (252, 518),
    },
    512: {
        1.000: (512, 512),
        1.333: (512, 384),
        1.524: (512, 336),
        1.778: (512, 288),
        2.000: (512, 256),
        3.200: (512, 160),
        0.750: (384, 512),
        0.656: (336, 512),
        0.562: (288, 512),
        0.500: (256, 512),
    },
    504: {
        1.000: (504, 504),
        1.333: (504, 378),
        1.565: (504, 322),
        1.800: (504, 280),
        2.118: (504, 238),
        3.273: (504, 154),
        0.750: (378, 504),
        0.639: (322, 504),
        0.556: (280, 504),
        0.472: (238, 504),
    },
}
RESIZE_MODES = ("fixed_mapping", "longest_side", "square", "fixed_size")

# A path or URL, a PIL image, or an (H, W, 3) RGB array (uint8 or float in [0, 1]).
ImageLike = Any


def _image_extensions() -> Tuple[str, ...]:
    """Folder image types; HEIC / HEIF when ``pillow_heif`` is installed."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return (".jpg", ".jpeg", ".png")
    register_heif_opener()
    return (".jpg", ".jpeg", ".png", ".heic", ".heif")


def decode_image(source: ImageLike):
    """Paths, URLs and PIL images -> uint8 (H, W, 3) numpy RGB with the EXIF
    orientation applied; arrays pass through."""
    from ...utils import load_image

    if isinstance(source, (str, Path)):
        source = load_image(str(source))
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"))
    return source


def read_image(source: ImageLike) -> mx.array:
    """RGB image -> uint8 (H, W, 3) MLX array. Float arrays in [0, 1] are
    quantized like the reference (truncated to uint8)."""
    image = decode_image(source)
    image = image if isinstance(image, mx.array) else mx.array(np.asarray(image))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) image, got {image.shape}")
    if image.dtype == mx.uint8:
        return image
    return mx.clip(image.astype(mx.float32) * 255, 0, 255).astype(mx.uint8)


def read_images(sources: Sequence[ImageLike]) -> List[mx.array]:
    """``read_image`` of each source, decoding them in parallel threads."""
    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as pool:
        return [read_image(image) for image in pool.map(decode_image, sources)]


def _round_uint8(x: mx.array) -> mx.array:
    return mx.clip(mx.floor(x + 0.5), 0, 255)


def resize_like_pil(image: mx.array, size: Tuple[int, int], shrink: bool) -> mx.array:
    """uint8 (H, W, 3) -> float32 (h, w, 3) in [0, 255], Pillow LANCZOS
    (``shrink``) or BICUBIC with Pillow's width-then-height passes."""
    x = image.astype(mx.float32)[None]
    height, width = image.shape[:2]
    if shrink:
        resize = resize_lanczos_nhwc
    else:
        resize = partial(resize_bicubic_nhwc, antialias=True)
    if size[1] != width:
        x = _round_uint8(resize(x, (height, size[1])))
    if size[0] != height:
        x = _round_uint8(resize(x, size))
    return x[0]


@lru_cache(maxsize=64)
def _nearest_rows(in_size: int, out_size: int) -> mx.array:
    """OpenCV ``INTER_NEAREST`` source indices."""
    step = 1.0 / (out_size / in_size)
    return mx.array([min(math.floor(i * step), in_size - 1) for i in range(out_size)])


class MapAnythingProcessor:
    """Resizes views to a common model resolution and normalizes the images.

    ``resize_mode``: ``fixed_mapping`` (closest aspect-ratio bucket of
    ``resolution_set``), ``longest_side`` / ``square`` (``size`` int) or
    ``fixed_size`` (``size`` = (width, height)); sizes snap to ``patch_size``.
    """

    def __init__(
        self,
        resize_mode: str = "fixed_mapping",
        size: Union[int, Sequence[int], None] = None,
        norm_type: str = "dinov2",
        patch_size: int = 14,
        resolution_set: int = 518,
        **kwargs,
    ):
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if resize_mode != "fixed_mapping" and size is None:
            raise ValueError(f"size is required for resize_mode={resize_mode!r}")
        if resize_mode in ("longest_side", "square") and not isinstance(size, int):
            raise ValueError(f"size must be an int for resize_mode={resize_mode!r}")
        if resize_mode == "fixed_size" and len(size) != 2:
            raise ValueError(
                "size must be (width, height) for resize_mode='fixed_size'"
            )
        if norm_type not in IMAGE_NORMALIZATIONS:
            raise ValueError(f"norm_type must be one of {list(IMAGE_NORMALIZATIONS)}")
        self.resize_mode = resize_mode
        self.size = size
        self.norm_type = norm_type
        self.patch_size = patch_size
        self.resolution_set = resolution_set
        mean, std = IMAGE_NORMALIZATIONS[norm_type]
        self.image_mean, self.image_std = mx.array(mean), mx.array(std)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_path = Path(path) / "preprocessor_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        config.update(kwargs)
        return cls(**config)

    def target_size(self, aspect_ratio: float) -> Tuple[int, int]:
        """Model input (width, height) for a mean aspect ratio (width / height)."""
        p = self.patch_size
        if self.resize_mode == "fixed_mapping":
            mapping = RESOLUTION_MAPPINGS[self.resolution_set]
            return mapping[min(mapping, key=lambda r: abs(r - aspect_ratio))]
        if self.resize_mode == "square":
            return (self.size // p) * p, (self.size // p) * p
        if self.resize_mode == "longest_side":
            if aspect_ratio >= 1:
                return self.size, round((self.size // p) / aspect_ratio) * p
            return round((self.size // p) * aspect_ratio) * p, self.size
        return (self.size[0] // p) * p, (self.size[1] // p) * p

    def normalize(self, image: mx.array) -> mx.array:
        """[0, 255] (..., 3) -> normalized float32 (1, H, W, 3)."""
        x = (image / 255.0 - self.image_mean) / self.image_std
        return x.reshape(-1, *x.shape[-3:])

    def crop_resize(
        self,
        image: mx.array,
        size: Tuple[int, int],
        depth: mx.array = None,
        intrinsics: mx.array = None,
    ):
        """``crop_resize_if_necessary``: scale so that the (width, height)
        ``size`` fits inside, then center-crop. Returns the float32 image in
        [0, 255] and the matching depth and intrinsics (None when absent)."""
        height, width = image.shape[:2]
        target_w, target_h = size
        scale = max(target_w / width, target_h / height) + 1e-8
        new_w, new_h = math.floor(width * scale), math.floor(height * scale)
        image = resize_like_pil(image, (new_h, new_w), shrink=scale < 1)
        if depth is not None:
            depth = mx.take(depth, _nearest_rows(height, new_h), axis=0)
            depth = mx.take(depth, _nearest_rows(width, new_w), axis=1)
        margin_x, margin_y = new_w - target_w, new_h - target_h
        if intrinsics is None:
            left, top = margin_x // 2, margin_y // 2
        else:
            # The reference derives the crop from the intrinsics: round half to even.
            left, top = round(margin_x / 2), round(margin_y / 2)
            shift_x = 0.5 * (width * scale - new_w) + 0.5 + left
            shift_y = 0.5 * (height * scale - new_h) + 0.5 + top
            half_pixel = mx.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
            shift = mx.array([[0.0, 0.0, shift_x], [0.0, 0.0, shift_y]])
            intrinsics = intrinsics.astype(mx.float32)
            rows = (intrinsics[:2] + half_pixel) * scale - shift
            intrinsics = mx.concatenate([rows, intrinsics[2:]])
        image = image[top : top + target_h, left : left + target_w]
        if depth is not None:
            depth = depth[top : top + target_h, left : left + target_w]
        return image, depth, intrinsics

    def preprocess(self, views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """``preprocess_inputs``: resize every view (image, ``depth_z``,
        ``intrinsics`` or ``ray_directions``) to the common model size and
        normalize the images; ``camera_poses`` and other keys pass through
        with a batch axis."""
        if not views:
            raise ValueError("views cannot be empty")
        images = read_images([view["img"] for view in views])
        size = self.target_size(np.mean([i.shape[1] / i.shape[0] for i in images]))
        processed = []
        for index, (view, image) in enumerate(zip(views, images)):
            if "intrinsics" in view and "ray_directions" in view:
                raise ValueError(
                    f"View {index} cannot have both 'intrinsics' and 'ray_directions'."
                )
            depth = view.get("depth_z")
            if depth is not None:
                depth = _as_array(depth)
                if depth.ndim != 2:
                    raise ValueError(
                        f"View {index}: depth_z must be (H, W), got {depth.shape}"
                    )
            intrinsics = view.get("intrinsics")
            if intrinsics is not None:
                intrinsics = _as_array(intrinsics)
            elif "ray_directions" in view:
                intrinsics = intrinsics_from_rays(_as_array(view["ray_directions"]))
            image, depth, intrinsics = self.crop_resize(image, size, depth, intrinsics)

            out = {"img": self.normalize(image), "data_norm_type": [self.norm_type]}
            if depth is not None:
                out["depth_z"] = depth[None]
            if intrinsics is not None:
                out["intrinsics"] = intrinsics[None]
            if "camera_poses" in view:
                poses = view["camera_poses"]
                if isinstance(poses, (tuple, list)):
                    out["camera_poses"] = tuple(_as_array(p)[None] for p in poses)
                else:
                    out["camera_poses"] = _as_array(poses)[None]
            for key, value in view.items():
                if key not in (
                    "img",
                    "depth_z",
                    "intrinsics",
                    "ray_directions",
                    "camera_poses",
                ):
                    out[key] = value
            processed.append(out)
        return processed

    def load_images(
        self, folder_or_list: Union[str, Sequence[ImageLike]], stride: int = 1
    ) -> List[Dict[str, Any]]:
        """``load_images``: a folder (sorted) or a list of images -> views of
        normalized images at the common model size."""
        if isinstance(folder_or_list, (str, Path)):
            root = Path(folder_or_list)
            entries = [root / name for name in sorted(os.listdir(root))]
        else:
            entries = list(folder_or_list)
        extensions = _image_extensions()
        sources = [
            entry
            for entry in entries[::stride]
            if not isinstance(entry, (str, Path))
            or str(entry).lower().endswith(extensions)
        ]
        if not sources:
            raise ValueError("No valid images found")
        views = self.preprocess([{"img": image} for image in sources])
        height, width = views[0]["img"].shape[1:3]
        for index, view in enumerate(views):
            view["true_shape"] = np.int32([[height, width]])
            view["idx"] = index
            view["instance"] = str(index)
        return views

    def __call__(self, inputs, **kwargs) -> List[Dict[str, Any]]:
        """Views (dicts) -> ``preprocess``; a folder or a list of images ->
        ``load_images``."""
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], dict):
            return self.preprocess(inputs)
        return self.load_images(inputs, **kwargs)


install_auto_processor_patch(["mapanything"], MapAnythingProcessor)
