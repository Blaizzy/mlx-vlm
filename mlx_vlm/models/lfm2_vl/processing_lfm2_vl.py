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
        self.do_image_splitting = False
        self.use_thumbnail = False
        self.max_num_patches = kwargs.get(
            "max_num_patches",
            self.max_image_tokens * self.downsample_factor**2,
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

    def preprocess(self, images, return_tensors=None, **kwargs):
        images = self._flatten_images(self.fetch_images(images))
        pixel_values = []
        pixel_attention_mask = []
        spatial_shapes = []

        for image in images:
            image = self._to_rgb_image(image)
            width, height = image.size

            if self.do_resize:
                target_width, target_height = _smart_resize(
                    height=height,
                    width=width,
                    downsample_factor=self.downsample_factor,
                    min_image_tokens=self.min_image_tokens,
                    max_image_tokens=self.max_image_tokens,
                    encoder_patch_size=self.encoder_patch_size,
                )
                image = image.resize(
                    (target_width, target_height), Image.Resampling.BILINEAR
                )
            else:
                target_width, target_height = width, height

            array = np.array(image, dtype=np.float32)
            if self.do_rescale:
                array *= self.rescale_factor
            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)
                std = np.array(self.image_std, dtype=np.float32)
                array = (array - mean) / std

            patches = _convert_image_to_patches(array, self.encoder_patch_size)
            h_patches = target_height // self.encoder_patch_size
            w_patches = target_width // self.encoder_patch_size

            if self.do_pad:
                patches, mask = _pad_along_first_dim(patches, self.max_num_patches)
            else:
                mask = np.ones((patches.shape[0],), dtype=np.int32)

            pixel_values.append(patches)
            pixel_attention_mask.append(mask)
            spatial_shapes.append((h_patches, w_patches))

        data = {
            "pixel_values": np.stack(pixel_values),
            "pixel_attention_mask": np.stack(pixel_attention_mask),
            "spatial_shapes": np.array(spatial_shapes, dtype=np.int32),
        }
        tensor_type = "np" if return_tensors == "np" else None
        return BatchFeature(data=data, tensor_type=tensor_type)

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


# Try to import the slow image processor to force its use. Some Transformers
# versions import torch from the SigLIP2 processor module, so fall back to a
# local PIL/NumPy implementation when torch/torchvision are absent.
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
    if "Fast" in processor_class_name and _SLOW_PROCESSOR_AVAILABLE:
        # Replace with slow processor using the same config
        if hasattr(image_processor, "to_dict"):
            # Use the config dict to create the slow processor
            slow_processor = Siglip2ImageProcessor(**image_processor.to_dict())
        else:
            # Fallback to copying attributes
            slow_processor = Siglip2ImageProcessor(
                **{
                    k: v
                    for k, v in image_processor.__dict__.items()
                    if not k.startswith("_") and k not in ["name_or_path"]
                }
            )
        image_processor = slow_processor

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
    if not hasattr(self.image_processor, "do_image_splitting"):
        self.image_processor.do_image_splitting = (
            False  # Disable tiling for slow processor
        )
    if not hasattr(self.image_processor, "use_thumbnail"):
        self.image_processor.use_thumbnail = False


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
        # The tokenizer is a standard PreTrainedTokenizerFast. Trusting remote
        # code here makes AutoConfig import modeling_lfm2_vl.py, which requires
        # torch even though MLX only needs the tokenizer files.
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
    # needs resizing enabled and no image splitting metadata.
    image_processor_config["do_resize"] = True
    image_processor_config["do_image_splitting"] = False

    image_processor = Siglip2ImageProcessor(**image_processor_config)
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
    Ensure we're using the slow image processor, not the fast one.
    The fast processor only supports PyTorch tensors which we can't use without PyTorch.
    """
    image_processor = processor_instance.image_processor
    processor_class_name = type(image_processor).__name__

    if "Fast" in processor_class_name and _SLOW_PROCESSOR_AVAILABLE:
        # Need to replace with slow processor
        # Get the config from the fast processor
        config = (
            image_processor.to_dict() if hasattr(image_processor, "to_dict") else {}
        )
        # Remove keys that might cause issues
        config.pop("image_processor_type", None)
        config.pop("auto_map", None)
        config.pop("_processor_class", None)

        # Create slow processor with the same config
        slow_processor = Siglip2ImageProcessor(**config)
        processor_instance.image_processor = slow_processor

        # Re-add missing attributes
        if not hasattr(processor_instance.image_processor, "tile_size"):
            processor_instance.image_processor.tile_size = 512
        if not hasattr(processor_instance.image_processor, "downsample_factor"):
            processor_instance.image_processor.downsample_factor = 2
        if not hasattr(processor_instance.image_processor, "do_image_splitting"):
            processor_instance.image_processor.do_image_splitting = False
        if not hasattr(processor_instance.image_processor, "use_thumbnail"):
            processor_instance.image_processor.use_thumbnail = False

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

    # Check if image_rows/cols/sizes are present (fast processor case)
    if "image_rows" in vision_inputs:
        image_rows = vision_inputs.pop("image_rows")
        image_cols = vision_inputs.pop("image_cols")
        image_sizes = vision_inputs.pop("image_sizes")
    else:
        # Slow processor case - compute from spatial_shapes or pixel_attention_mask
        # The spatial_shapes gives the actual (height, width) in patches for each image
        spatial_shapes = vision_inputs.get("spatial_shapes")
        if spatial_shapes is not None:
            # spatial_shapes is array of shape (batch, 2) with [height, width] in patches
            image_rows = [[int(ss[0])] for ss in spatial_shapes]
            image_cols = [[int(ss[1])] for ss in spatial_shapes]
            image_sizes = [[int(ss[0] * ss[1])] for ss in spatial_shapes]
        else:
            # Fallback to computing from pixel_values
            pixel_values = vision_inputs.get("pixel_values")
            patch_size = getattr(self.image_processor, "patch_size", 16)
            image_rows, image_cols, image_sizes = _compute_image_grid_info(
                pixel_values, patch_size
            )

    # For slow processor, use simplified text expansion
    # (no tiling support, just add image tokens)
    # Account for downsample_factor: the model pads odd patch-grid dimensions
    # before downsampling, so the token count is ceil(rows/f)*ceil(cols/f).
    downsample_factor = getattr(self.image_processor, "downsample_factor", 2)

    expanded_text = []
    for sample_text, sample_images, rows, cols, _sizes in zip(
        text, batched_images, image_rows, image_cols, image_sizes
    ):
        rows = _normalize_image_layout_axis(rows, len(sample_images))
        cols = _normalize_image_layout_axis(cols, len(sample_images))
        split_sample = sample_text.split(self.image_token)
        result = ""
        for i, _ in enumerate(sample_images):
            result += split_sample[i]
            if use_image_special_tokens:
                result += self.image_start_token
            # Add image tokens based on the number of patches AFTER downsampling
            # The model pads odd patch-grid dimensions before downsampling.
            # Use rows/cols (patch grid) rather than total patches to mirror it.
            num_rows = rows[i] if i < len(rows) else rows[0]
            num_cols = cols[i] if i < len(cols) else cols[0]
            num_image_tokens = _num_image_tokens_from_patch_grid(
                int(num_rows), int(num_cols), downsample_factor
            )
            result += self.image_token * num_image_tokens
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
