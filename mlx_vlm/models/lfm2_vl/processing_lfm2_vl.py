"""
Compatibility patch for Lfm2VlProcessor.

The Lfm2VlProcessorKwargs has a default `return_row_col_info: True` in images_kwargs,
but this parameter is only supported by the FAST image processor (Lfm2VlImageProcessorFast).
When using the slow image processor (Siglip2ImageProcessor), this causes a validation error.

This patch:
1. Removes the unsupported `return_row_col_info` parameter from the defaults
2. Enables `do_resize: True` to ensure images are properly resized for patch processing
3. Patches the `__call__` method to handle the slow image processor case, computing
   `image_rows`, `image_cols`, `image_sizes` when missing and providing sensible
   defaults for tile-related parameters
4. Patches the `__init__` to add missing attributes to the slow image processor
5. Forces the use of the slow image processor to avoid PyTorch tensor requirements
6. Implements the official image splitting (tiling) in the numpy image processor:
   large images are split into a grid of `tile_size` tiles (plus a downsampled
   thumbnail) and the text is expanded with the official per-tile
   `<|img_row_r_col_c|>` / `<|img_thumbnail|>` marker tokens
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import ImageProcessingMixin
from transformers.models.lfm2_vl.processing_lfm2_vl import (
    Lfm2VlProcessor,
    Lfm2VlProcessorKwargs,
)

from ..base import install_auto_processor_patch, load_chat_template

# Official transformers preprocessing defaults for LFM2-VL: images larger than
# `max_pixels_tolerance` * `max_image_tokens` * patch^2 * factor^2 pixels are
# split into a grid of `tile_size` tiles (grid within [min_tiles, max_tiles],
# aspect ratio preserved) plus a downsampled thumbnail appended last.
# LiquidAI's MLX repos ship `do_image_splitting: false` / `use_thumbnail: false`
# — a handicap left over from the pre-tiling numpy processor that notably
# hurts grounding accuracy on large screenshots — so these defaults are
# re-applied when loading unless the user overrides them explicitly (via
# `from_pretrained` kwargs or per-call processor kwargs).
_OFFICIAL_SPLITTING_DEFAULTS = {
    "do_image_splitting": True,
    "min_tiles": 2,
    "max_tiles": 10,
    "use_thumbnail": True,
}


def _num_image_tokens_from_patch_grid(
    rows: int, cols: int, downsample_factor: int
) -> int:
    """
    Compute the number of <image> placeholder tokens expected by the model.

    The LFM2-VL model downsamples the patch grid via PixelUnshuffleBlock. That
    block pads odd patch-grid dimensions up to the next multiple of
    `downsample_factor` before downsampling. The text expansion must mirror
    that padding behavior to keep image token count aligned with the produced
    image embeddings.
    """
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be a positive integer")

    padded_rows = rows + (-rows % downsample_factor)
    padded_cols = cols + (-cols % downsample_factor)
    return (padded_rows // downsample_factor) * (padded_cols // downsample_factor)


def _normalize_image_layout_axis(values, num_images: int) -> list[int]:
    """Normalize scalar or array-like row/col metadata to a per-image list."""
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return [int(values.item())] * max(1, num_images)
        return [int(v) for v in values.tolist()]

    if np.isscalar(values):
        return [int(values)] * max(1, num_images)

    return [int(v) for v in values]


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    downsample_factor: int,
    min_image_tokens: int,
    max_image_tokens: int,
    encoder_patch_size: int,
) -> tuple[int, int]:
    total_factor = encoder_patch_size * downsample_factor
    min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
    max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

    h_bar = max(total_factor, _round_by_factor(height, total_factor))
    w_bar = max(total_factor, _round_by_factor(width, total_factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(
            total_factor,
            math.floor(height / beta / total_factor) * total_factor,
        )
        w_bar = max(
            total_factor,
            math.floor(width / beta / total_factor) * total_factor,
        )
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    return w_bar, h_bar


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """
    Find the target aspect ratio closest to the given one.

    Ties are broken in favor of the ratio whose tile area best matches the
    original image area, mirroring the official LFM2-VL image processor.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        # update best ratio if we found a closer match
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # if equally close, prefer the ratio that better matches the original area
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


def _target_tile_ratios(min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
    """All (width, height) tile grids whose tile count fits the budget."""
    ratios = [
        (w, h)
        for n in range(min_tiles, max_tiles + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if min_tiles <= w * h <= max_tiles
    ]
    return sorted(set(ratios), key=lambda x: x[0] * x[1])


def _get_grid_layout(
    height: int,
    width: int,
    min_tiles: int,
    max_tiles: int,
    tile_size: int,
) -> tuple[int, int, int, int]:
    """
    Pick the tile grid for an image: (grid_width, grid_height, target_w, target_h).

    The grid preserves the image aspect ratio as closely as possible while
    keeping the tile count within [min_tiles, max_tiles].
    """
    aspect_ratio = width / height
    target_ratios = _target_tile_ratios(min_tiles, max_tiles)
    grid_width, grid_height = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, tile_size
    )
    return (
        grid_width,
        grid_height,
        tile_size * grid_width,
        tile_size * grid_height,
    )


def _is_image_too_large(
    height: int,
    width: int,
    max_image_tokens: int,
    encoder_patch_size: int,
    downsample_factor: int,
    max_pixels_tolerance: float,
) -> bool:
    """Check if the image is too large to be processed as a single tile."""
    total_factor = encoder_patch_size * downsample_factor

    h_bar = max(encoder_patch_size, _round_by_factor(height, total_factor))
    w_bar = max(encoder_patch_size, _round_by_factor(width, total_factor))
    return (
        h_bar * w_bar
        > max_image_tokens
        * encoder_patch_size**2
        * downsample_factor**2
        * max_pixels_tolerance
    )


def _convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width, channels = image.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    image = image[: num_patches_height * patch_size, : num_patches_width * patch_size]
    patches = image.reshape(
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(num_patches_height * num_patches_width, -1)


def _pad_along_first_dim(
    array: np.ndarray, target_length: int
) -> tuple[np.ndarray, np.ndarray]:
    current_length = array.shape[0]
    mask = np.ones((target_length,), dtype=np.int32)
    if current_length >= target_length:
        return array[:target_length], mask

    pad_shape = (target_length - current_length,) + array.shape[1:]
    padded = np.concatenate(
        [array, np.zeros(pad_shape, dtype=array.dtype)],
        axis=0,
    )
    mask[current_length:] = 0
    return padded, mask


class Lfm2VlNumpyImageProcessor(ImageProcessingMixin):
    """PIL/NumPy image processor compatible with the LFM2-VL packed-patch input."""

    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(self, **kwargs):
        self.image_mean = kwargs.get("image_mean", [0.5, 0.5, 0.5])
        self.image_std = kwargs.get("image_std", [0.5, 0.5, 0.5])
        self.rescale_factor = kwargs.get("rescale_factor", 1 / 255)
        self.do_rescale = kwargs.get("do_rescale", True)
        self.do_normalize = kwargs.get("do_normalize", True)
        self.do_resize = kwargs.get("do_resize", True)
        self.do_pad = kwargs.get("do_pad", True)
        self.downsample_factor = kwargs.get("downsample_factor", 2)
        self.encoder_patch_size = kwargs.get(
            "encoder_patch_size", kwargs.get("patch_size", 16)
        )
        self.patch_size = self.encoder_patch_size
        self.min_image_tokens = kwargs.get("min_image_tokens", 64)
        self.max_image_tokens = kwargs.get("max_image_tokens", 256)
        self.tile_size = kwargs.get("tile_size", 512)
        self.max_pixels_tolerance = kwargs.get("max_pixels_tolerance", 2.0)
        for key, value in _OFFICIAL_SPLITTING_DEFAULTS.items():
            setattr(self, key, kwargs.get(key, value))
        # Each vision-tower row holds one tile or the thumbnail; both must fit
        # within the padded patch budget.
        tile_size_patches = (
            (self.tile_size // self.encoder_patch_size) ** 2
            if self.do_image_splitting
            else 0
        )
        self.max_num_patches = kwargs.get(
            "max_num_patches",
            max(self.max_image_tokens * self.downsample_factor**2, tile_size_patches),
        )

    def fetch_images(self, images):
        if isinstance(images, (list, tuple)):
            return [self.fetch_images(image) for image in images]
        if isinstance(images, (str, Path)):
            return Image.open(images)
        return images

    def _to_rgb_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _flatten_images(self, images):
        if isinstance(images, (list, tuple)):
            flattened = []
            for image in images:
                flattened.extend(self._flatten_images(image))
            return flattened
        return [images]

    def _views_for_image(
        self,
        image: Image.Image,
        do_image_splitting: bool,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
    ) -> tuple[list[Image.Image], int, int, tuple[int, int]]:
        """
        Build the vision-tower views for one source image.

        Large images are resized to a `tile_size` grid and split into tiles
        (row-major), with a smart-resized thumbnail appended last. Smaller
        images yield the single smart-resized view, exactly as before.

        Returns (views, grid_rows, grid_cols, resized_size) where
        `resized_size` is the (height, width) used for the single view and the
        thumbnail.
        """
        width, height = image.size
        resized_width, resized_height = _smart_resize(
            height=height,
            width=width,
            downsample_factor=self.downsample_factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
            encoder_patch_size=self.encoder_patch_size,
        )
        is_image_large = _is_image_too_large(
            height=height,
            width=width,
            max_image_tokens=self.max_image_tokens,
            encoder_patch_size=self.encoder_patch_size,
            downsample_factor=self.downsample_factor,
            max_pixels_tolerance=self.max_pixels_tolerance,
        )

        if self.do_resize and do_image_splitting and is_image_large:
            grid_width, grid_height, target_width, target_height = _get_grid_layout(
                height, width, min_tiles, max_tiles, self.tile_size
            )
            resized = image.resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
            tile = self.tile_size
            views = [
                resized.crop(
                    (col * tile, row * tile, (col + 1) * tile, (row + 1) * tile)
                )
                for row in range(grid_height)
                for col in range(grid_width)
            ]
            if use_thumbnail and grid_width * grid_height > 1:
                views.append(
                    image.resize(
                        (resized_width, resized_height), Image.Resampling.BILINEAR
                    )
                )
            return views, grid_height, grid_width, (resized_height, resized_width)

        if self.do_resize:
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )
            return [image], 1, 1, (resized_height, resized_width)

        return [image], 1, 1, (height, width)

    def _view_to_patches(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Rescale/normalize one view and pack it into flattened patch rows."""
        width, height = image.size

        array = np.array(image, dtype=np.float32)
        if self.do_rescale:
            array *= self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)
            std = np.array(self.image_std, dtype=np.float32)
            array = (array - mean) / std

        patches = _convert_image_to_patches(array, self.encoder_patch_size)
        h_patches = height // self.encoder_patch_size
        w_patches = width // self.encoder_patch_size

        if self.do_pad:
            patches, mask = _pad_along_first_dim(patches, self.max_num_patches)
        else:
            mask = np.ones((patches.shape[0],), dtype=np.int32)

        return patches, mask, h_patches, w_patches

    def preprocess(self, images, return_tensors=None, **kwargs):
        images = self._flatten_images(self.fetch_images(images))

        # Per-call overrides win over the processor defaults.
        do_image_splitting = bool(
            kwargs.get("do_image_splitting", self.do_image_splitting)
        )
        min_tiles = int(kwargs.get("min_tiles", self.min_tiles))
        max_tiles = int(kwargs.get("max_tiles", self.max_tiles))
        use_thumbnail = bool(kwargs.get("use_thumbnail", self.use_thumbnail))

        if do_image_splitting and min_tiles > max_tiles:
            raise ValueError("min_tiles must be less than or equal to max_tiles")

        pixel_values = []
        pixel_attention_mask = []
        spatial_shapes = []
        # Official layout metadata per source image: tile-grid dims and the
        # smart-resized (height, width), used to expand the text placeholders.
        image_rows = []
        image_cols = []
        image_sizes = []

        for image in images:
            image = self._to_rgb_image(image)
            views, rows, cols, resized_size = self._views_for_image(
                image, do_image_splitting, min_tiles, max_tiles, use_thumbnail
            )
            for view in views:
                patches, mask, h_patches, w_patches = self._view_to_patches(view)
                pixel_values.append(patches)
                pixel_attention_mask.append(mask)
                spatial_shapes.append((h_patches, w_patches))
            image_rows.append(rows)
            image_cols.append(cols)
            image_sizes.append(list(resized_size))

        data = {
            "pixel_values": np.stack(pixel_values),
            "pixel_attention_mask": np.stack(pixel_attention_mask),
            "spatial_shapes": np.array(spatial_shapes, dtype=np.int32),
            "image_rows": np.array(image_rows, dtype=np.int32),
            "image_cols": np.array(image_cols, dtype=np.int32),
            "image_sizes": np.array(image_sizes, dtype=np.int32),
        }
        tensor_type = "np" if return_tensors == "np" else None
        return BatchFeature(data=data, tensor_type=tensor_type)

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


# Try to import the slow image processor. Some Transformers versions import
# torch from the SigLIP2 processor module. Historically the real Siglip2
# processor was used when importable, but it has no LFM2-VL tiling and its
# (B, C, H, W) output never matched the packed-patch contract the MLX model
# consumes — so behavior silently differed between torch and torch-free
# environments. The NumPy processor below is now used unconditionally; this
# import only feeds the _SLOW_PROCESSOR_AVAILABLE fallback below.
try:
    from transformers.models.siglip2.image_processing_siglip2 import (
        Siglip2ImageProcessor,
    )

    _SLOW_PROCESSOR_AVAILABLE = True
except ImportError:
    Siglip2ImageProcessor = Lfm2VlNumpyImageProcessor
    _SLOW_PROCESSOR_AVAILABLE = True

# Remove return_row_col_info from the defaults since the slow image processor
# (Siglip2ImageProcessor) doesn't support it - only the fast version does.
# Also enable do_resize to ensure images are properly resized to be divisible by patch_size.
if hasattr(Lfm2VlProcessorKwargs, "_defaults"):
    if "images_kwargs" in Lfm2VlProcessorKwargs._defaults:
        Lfm2VlProcessorKwargs._defaults["images_kwargs"].pop(
            "return_row_col_info", None
        )
        # Enable resizing for the slow image processor (model config has do_resize: False
        # which is intended for the fast processor that handles resizing differently)
        Lfm2VlProcessorKwargs._defaults["images_kwargs"]["do_resize"] = True


# Store the original __init__ method
_original_init = Lfm2VlProcessor.__init__


def _patched_init(self, image_processor, tokenizer, chat_template=None, **kwargs):
    """Patched __init__ that adds missing attributes to the slow image processor."""
    # Check if we got the fast image processor and need to replace it with the slow one
    # The fast processor requires PyTorch tensors which we don't have
    processor_class_name = type(image_processor).__name__
    # Always swap in the NumPy image processor: it is the only implementation
    # with the official LFM2-VL tiling + thumbnail and the packed-patch output
    # the MLX model consumes. The real Siglip2ImageProcessor (instantiated
    # when torch is installed) has neither.
    if processor_class_name != "Lfm2VlNumpyImageProcessor":
        # Replace with the NumPy processor using the same config
        if hasattr(image_processor, "to_dict"):
            # Use the config dict to create the replacement
            config = image_processor.to_dict()
        else:
            # Fallback to copying attributes
            config = {
                k: v
                for k, v in image_processor.__dict__.items()
                if not k.startswith("_") and k not in ["name_or_path"]
            }
        # Apply the official tiling defaults (see _OFFICIAL_SPLITTING_DEFAULTS):
        # MLX repos ship tiling disabled, which the NumPy processor can now do.
        for key in _OFFICIAL_SPLITTING_DEFAULTS:
            config.pop(key, None)
        image_processor = Lfm2VlNumpyImageProcessor(**config)

    # Call original __init__
    _original_init(
        self, image_processor, tokenizer, chat_template=chat_template, **kwargs
    )

    # Add missing attributes for the slow image processor (Siglip2ImageProcessor)
    # These are needed by expand_text_with_placeholders and _get_image_num_tokens
    if not hasattr(self.image_processor, "tile_size"):
        self.image_processor.tile_size = 512
    if not hasattr(self.image_processor, "max_image_tokens"):
        self.image_processor.max_image_tokens = 256
    if not hasattr(self.image_processor, "min_image_tokens"):
        self.image_processor.min_image_tokens = 64
    if not hasattr(self.image_processor, "downsample_factor"):
        self.image_processor.downsample_factor = 2
    if not hasattr(self.image_processor, "encoder_patch_size"):
        self.image_processor.encoder_patch_size = 16
    for key, value in _OFFICIAL_SPLITTING_DEFAULTS.items():
        if not hasattr(self.image_processor, key):
            setattr(self.image_processor, key, value)


# Apply the __init__ patch
Lfm2VlProcessor.__init__ = _patched_init

_original_from_pretrained = Lfm2VlProcessor.from_pretrained


@classmethod
def _patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load LFM2-VL with the slow Siglip2 image processor to avoid torch/torchvision."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    kwargs.pop("trust_remote_code", None)
    kwargs.pop("use_fast", None)

    model_path = Path(pretrained_model_name_or_path)
    is_local = model_path.exists() and model_path.is_dir()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path) if is_local else pretrained_model_name_or_path,
        trust_remote_code=False,
        local_files_only=is_local,
    )
    if is_local:
        load_chat_template(tokenizer, model_path)

    if not _SLOW_PROCESSOR_AVAILABLE:
        return _original_from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )

    if is_local:
        config_path = model_path / "processor_config.json"
        if not config_path.exists():
            config_path = model_path / "preprocessor_config.json"
    else:
        try:
            config_path = Path(
                hf_hub_download(pretrained_model_name_or_path, "processor_config.json")
            )
        except Exception:
            config_path = Path(
                hf_hub_download(
                    pretrained_model_name_or_path, "preprocessor_config.json"
                )
            )

    image_processor_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            image_processor_config = json.load(f)
        image_processor_config = image_processor_config.get(
            "image_processor", image_processor_config
        )

    for key in (
        "image_processor_type",
        "processor_class",
        "image_seq_length",
        "return_row_col_info",
        "device",
        "disable_grouping",
        "return_tensors",
        "input_data_format",
    ):
        image_processor_config.pop(key, None)

    # The upstream config is tuned for the fast processor; the slow Siglip2 path
    # needs resizing enabled.
    image_processor_config["do_resize"] = True

    # Re-apply the official tiling defaults (see _OFFICIAL_SPLITTING_DEFAULTS):
    # the LiquidAI MLX repos ship `do_image_splitting: false` / `use_thumbnail:
    # false`, which disables the high-resolution splitting the model was trained
    # and evaluated with. Users can still opt out explicitly, e.g.
    # `Lfm2VlProcessor.from_pretrained(model, do_image_splitting=False)`.
    for key, default in _OFFICIAL_SPLITTING_DEFAULTS.items():
        image_processor_config.pop(key, None)
        image_processor_config[key] = kwargs.pop(key, default)

    # The NumPy image processor is used unconditionally (torch or not): it is
    # the only implementation with official tiling + the packed-patch output.
    image_processor = Lfm2VlNumpyImageProcessor(**image_processor_config)
    return cls(image_processor=image_processor, tokenizer=tokenizer)


def _compute_image_grid_info(pixel_values, patch_size: int = 16):
    """
    Compute image_rows, image_cols, and image_sizes from pixel_values.

    When using the slow image processor (Siglip2ImageProcessor), these values
    are not returned. This function computes them from the pixel_values tensor.

    Args:
        pixel_values: Array of shape (batch, num_patches, patch_dim)
        patch_size: The patch size used for image processing

    Returns:
        image_rows: List of rows per image
        image_cols: List of cols per image
        image_sizes: List of total patches per image
    """
    # pixel_values shape: (batch, num_patches, patch_dim)
    # For Siglip2, each image is processed independently and has its own num_patches
    if hasattr(pixel_values, "shape"):
        batch_size = pixel_values.shape[0]
        num_patches = pixel_values.shape[1]

        # Estimate rows/cols from num_patches (assuming roughly square)
        # The actual image was resized to fit max_num_patches while maintaining aspect ratio
        side_length = int(math.sqrt(num_patches))

        # Return as nested lists (one list per batch, one value per image in batch)
        image_rows = [[side_length] for _ in range(batch_size)]
        image_cols = [[side_length] for _ in range(batch_size)]
        image_sizes = [[num_patches] for _ in range(batch_size)]

        return image_rows, image_cols, image_sizes

    return [[1]], [[1]], [[1]]


# Store the original __call__ method
_original_call = Lfm2VlProcessor.__call__


def _ensure_slow_processor(processor_instance):
    """
    Ensure we're using the NumPy image processor.

    The fast (torchvision) processor needs PyTorch tensors and the real slow
    Siglip2ImageProcessor has no LFM2-VL tiling — neither matches the packed
    patch input the MLX model expects, so both are swapped for the NumPy one.
    """
    image_processor = processor_instance.image_processor
    processor_class_name = type(image_processor).__name__

    if processor_class_name != "Lfm2VlNumpyImageProcessor":
        # Get the config from the existing processor
        config = (
            image_processor.to_dict() if hasattr(image_processor, "to_dict") else {}
        )
        # Remove keys that might cause issues
        config.pop("image_processor_type", None)
        config.pop("auto_map", None)
        config.pop("_processor_class", None)
        # Apply the official tiling defaults (see _OFFICIAL_SPLITTING_DEFAULTS):
        # MLX repos ship tiling disabled, which the NumPy processor can now do.
        for key in _OFFICIAL_SPLITTING_DEFAULTS:
            config.pop(key, None)

        # Create the NumPy processor with the same config
        numpy_processor = Lfm2VlNumpyImageProcessor(**config)
        processor_instance.image_processor = numpy_processor

        # Re-add missing attributes
        if not hasattr(processor_instance.image_processor, "tile_size"):
            processor_instance.image_processor.tile_size = 512
        if not hasattr(processor_instance.image_processor, "downsample_factor"):
            processor_instance.image_processor.downsample_factor = 2
        for key, value in _OFFICIAL_SPLITTING_DEFAULTS.items():
            if not hasattr(processor_instance.image_processor, key):
                setattr(processor_instance.image_processor, key, value)

    return processor_instance.image_processor


def _patched_call(self, images=None, text=None, **kwargs):
    """
    Patched __call__ that handles the slow image processor case.

    The slow Siglip2ImageProcessor doesn't return image_rows, image_cols, image_sizes
    which are required by expand_text_with_placeholders. This patch intercepts the call
    and computes these values when they're missing.
    """
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.image_utils import make_nested_list_of_images

    # Allow explicit per-call tiling overrides (do_image_splitting, min_tiles,
    # max_tiles, use_thumbnail) regardless of how the processor's kwarg
    # routing classifies them.
    splitting_overrides = {
        key: kwargs.pop(key) for key in _OFFICIAL_SPLITTING_DEFAULTS if key in kwargs
    }

    # Ensure we're using the slow processor (fast requires PyTorch tensors)
    if images is not None:
        _ensure_slow_processor(self)

    if images is None and text is not None:
        # Text-only case
        output_kwargs = self._merge_kwargs(
            Lfm2VlProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["text_kwargs"].pop("use_image_special_tokens", None)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(text_inputs, tensor_type=return_tensors)

    if text is None and images is None:
        raise ValueError("You must provide one of `text` or `images`.")

    if images is not None and text is None:
        raise ValueError(
            "You must provide `text` when `images` is provided. Minimal text consists of a single image token."
        )

    # Merge kwargs to get the final settings
    output_kwargs = self._merge_kwargs(
        Lfm2VlProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )
    output_kwargs["images_kwargs"].update(splitting_overrides)

    if isinstance(text, str):
        text = [text]
    elif text is not None and not isinstance(text, list):
        raise TypeError(
            "Invalid input text. Please provide a string, or a list of strings"
        )

    n_images_in_text = [sample.count(self.image_token) for sample in text]

    inputs = {}
    use_image_special_tokens = output_kwargs["text_kwargs"].pop(
        "use_image_special_tokens", True
    )

    # Process images
    images = self.image_processor.fetch_images(images)
    batched_images = make_nested_list_of_images(images)

    # Override return_tensors for image processing to avoid PyTorch dependency
    images_kwargs = output_kwargs["images_kwargs"].copy()
    images_kwargs["return_tensors"] = "np"  # Use numpy instead of pt

    vision_inputs = self.image_processor(batched_images, **images_kwargs)

    n_images_in_images = [len(sublist) for sublist in batched_images]
    if n_images_in_images != n_images_in_text:
        raise ValueError(
            f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
        )

    # Check if image_rows/cols/sizes are present (numpy processor with tiling)
    if "image_rows" in vision_inputs:
        image_rows = vision_inputs.pop("image_rows")
        image_cols = vision_inputs.pop("image_cols")
        image_sizes = vision_inputs.pop("image_sizes")
    else:
        # Image processors without layout metadata (e.g. the slow Siglip2 one):
        # every row of spatial_shapes is a single-tile image, so derive the
        # per-image pixel size from its patch grid.
        patch_size = getattr(self.image_processor, "patch_size", 16)
        spatial_shapes = vision_inputs.get("spatial_shapes")
        if spatial_shapes is not None:
            image_sizes = [
                (int(ss[0]) * patch_size, int(ss[1]) * patch_size)
                for ss in spatial_shapes
            ]
        else:
            # Fallback to estimating from pixel_values
            pixel_values = vision_inputs.get("pixel_values")
            grid_rows, grid_cols, _ = _compute_image_grid_info(pixel_values, patch_size)
            image_sizes = [
                (int(rows[0]) * patch_size, int(cols[0]) * patch_size)
                for rows, cols in zip(grid_rows, grid_cols)
            ]
        image_rows = [1] * len(image_sizes)
        image_cols = [1] * len(image_sizes)

    n_images = sum(len(sublist) for sublist in batched_images)
    flat_rows = _normalize_image_layout_axis(image_rows, n_images)
    flat_cols = _normalize_image_layout_axis(image_cols, n_images)
    flat_sizes = list(image_sizes)

    # Mirror the official Lfm2VlProcessor text expansion: a multi-tile image
    # becomes one section of row/col-marked tiles (row-major) followed by the
    # thumbnail, each tile contributing ceil(tile_patches / f)^2 tokens after
    # the pixel-unshuffle downsampling.
    downsample_factor = getattr(self.image_processor, "downsample_factor", 2)
    encoder_patch_size = getattr(self.image_processor, "encoder_patch_size", 16)
    tile_size = getattr(self.image_processor, "tile_size", 512)
    tile_patches = tile_size // encoder_patch_size
    tokens_per_tile = _num_image_tokens_from_patch_grid(
        tile_patches, tile_patches, downsample_factor
    )
    use_thumbnail = output_kwargs["images_kwargs"].get(
        "use_thumbnail", getattr(self.image_processor, "use_thumbnail", False)
    )
    image_thumbnail_token = getattr(self, "image_thumbnail_token", "<|img_thumbnail|>")

    expanded_text = []
    image_idx = 0
    for sample_text, sample_images in zip(text, batched_images):
        split_sample = sample_text.split(self.image_token)
        result = ""
        for i in range(len(sample_images)):
            result += split_sample[i]
            rows = int(flat_rows[image_idx])
            cols = int(flat_cols[image_idx])
            image_height, image_width = flat_sizes[image_idx]
            image_idx += 1

            # Tokens for the resized image (single-tile) or the thumbnail,
            # accounting for the pixel-unshuffle padding of odd patch grids.
            tokens_for_image = _num_image_tokens_from_patch_grid(
                image_height // encoder_patch_size,
                image_width // encoder_patch_size,
                downsample_factor,
            )

            if use_image_special_tokens:
                result += self.image_start_token
            if rows > 1 or cols > 1:
                for row in range(rows):
                    for col in range(cols):
                        if use_image_special_tokens:
                            result += f"<|img_row_{row + 1}_col_{col + 1}|>"
                        result += self.image_token * tokens_per_tile
                if use_thumbnail:
                    if use_image_special_tokens:
                        result += image_thumbnail_token
                    result += self.image_token * tokens_for_image
            else:
                result += self.image_token * tokens_for_image
            if use_image_special_tokens:
                result += self.image_end_token
        # Add any remaining text after the last image
        if len(split_sample) > len(sample_images):
            result += split_sample[-1]
        expanded_text.append(result)

    inputs.update(vision_inputs)

    return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

    text_inputs = self.tokenizer(expanded_text, **output_kwargs["text_kwargs"])
    inputs.update(text_inputs)

    # Convert lists to numpy arrays for proper handling by mlx_vlm
    # The tokenizer returns lists but mlx_vlm expects numpy arrays
    if isinstance(inputs.get("input_ids"), list):
        inputs["input_ids"] = np.array(inputs["input_ids"])
    if isinstance(inputs.get("attention_mask"), list):
        inputs["attention_mask"] = np.array(inputs["attention_mask"])

    return BatchFeature(
        inputs, tensor_type=None
    )  # Don't convert, let mlx_vlm handle it


# Apply the patch
Lfm2VlProcessor.from_pretrained = _patched_from_pretrained
Lfm2VlProcessor.__call__ = _patched_call

install_auto_processor_patch(["lfm2_vl", "lfm2-vl"], Lfm2VlProcessor)
