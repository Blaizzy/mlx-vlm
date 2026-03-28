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


# Try to import the slow image processor to force its use
try:
    from transformers.models.siglip2.image_processing_siglip2 import (
        Siglip2ImageProcessor,
    )

    _SLOW_PROCESSOR_AVAILABLE = True
except ImportError:
    _SLOW_PROCESSOR_AVAILABLE = False

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
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if is_local:
        load_chat_template(tokenizer, model_path)

    if not _SLOW_PROCESSOR_AVAILABLE:
        return _original_from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )

    config_path = (
        model_path / "preprocessor_config.json"
        if is_local
        else Path(
            hf_hub_download(pretrained_model_name_or_path, "preprocessor_config.json")
        )
    )

    image_processor_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            image_processor_config = json.load(f)

    for key in (
        "image_processor_type",
        "processor_class",
        "image_seq_length",
        "return_row_col_info",
        "device",
        "disable_grouping",
        "return_tensors",
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
