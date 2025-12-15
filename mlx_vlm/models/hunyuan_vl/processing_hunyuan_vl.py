"""Image processor and processor classes for HunyuanVL.

Based on the official HuggingFace transformers implementation.
Handles image preprocessing and tokenization for the HunyuanVL model.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

# CLIP normalization constants (same as HF)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
) -> Tuple[int, int]:
    """Rescale image dimensions to meet constraints.

    Ensures:
    1. Both dimensions are divisible by 'factor' (patch_size * merge_size)
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        factor: Divisibility factor (default: patch_size * merge_size = 16 * 2 = 32)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (resized_height, resized_width)
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # Round to nearest factor
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        # Scale down to fit max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        # Scale up to meet min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class HunYuanVLImageProcessor(ImageProcessingMixin):
    """Image processor for HunyuanVL model.

    Handles resizing, normalization, and patch extraction for images.

    Note: This class inherits from ImageProcessingMixin but NOT BaseImageProcessor
    to avoid automatic prepare_inputs() behavior while satisfying type checks.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        min_pixels: int = 512 * 512,
        max_pixels: int = 2048 * 2048,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
        image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
        do_resize: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Override with config values if provided
        if config is not None:
            vision_config = config.get("vision_config", {})
            min_pixels = config.get("min_pixels", min_pixels)
            max_pixels = config.get("max_pixels", max_pixels)
            patch_size = vision_config.get("patch_size", patch_size)
            merge_size = vision_config.get("spatial_merge_size", merge_size)

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    def _preprocess_single(
        self,
        image: Image.Image,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Preprocess a single image.

        Args:
            image: PIL Image

        Returns:
            Tuple of (flattened_patches, (grid_t, grid_h, grid_w))
        """
        # Convert to RGB if needed
        if self.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        resized_width, resized_height = width, height

        # Resize to meet constraints
        if self.do_resize:
            factor = self.patch_size * self.merge_size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image = image.resize((resized_width, resized_height), Image.BILINEAR)

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        if self.do_normalize:
            mean = np.array(self.image_mean).reshape(1, 1, 3)
            std = np.array(self.image_std).reshape(1, 1, 3)
            img_array = (img_array - mean) / std

        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)  # (C, H, W)

        # Calculate grid dimensions
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        grid_t = 1  # temporal dimension (always 1 for images)

        # Reshape to patches
        # Shape: (C, H, W) -> (C, grid_h, merge_size, patch_size, grid_w, merge_size, patch_size)
        channel = img_array.shape[0]
        patches = img_array.reshape(
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Transpose and flatten
        # Target: (num_patches, C * patch_size * patch_size)
        patches = patches.transpose(
            1, 2, 4, 5, 0, 3, 6
        )  # (gh/m, m, gw/m, m, C, ps, ps)
        flatten_patches = patches.reshape(
            grid_h * grid_w,
            channel * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Preprocess one or more images.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            Dictionary with:
                - pixel_values: (total_patches, C * patch_size * patch_size)
                - image_grid_thw: (num_images, 3) with [temporal, height, width] grids
        """
        if isinstance(images, Image.Image):
            images = [images]

        all_patches = []
        all_grids = []

        for image in images:
            patches, grid_thw = self._preprocess_single(image)
            all_patches.append(patches)
            all_grids.append(grid_thw)

        # Stack patches from all images
        pixel_values = np.concatenate(all_patches, axis=0)
        image_grid_thw = np.array(all_grids)

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        **kwargs,
    ) -> BatchFeature:
        """Process images and return BatchFeature."""
        data = self.preprocess(images, **kwargs)
        return BatchFeature(data=data)

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        **kwargs,
    ) -> int:
        """Calculate number of image tokens for given dimensions.

        Token count formula: patch_h * (patch_w + 1) + 2
        - patch_h = grid_h / merge_size
        - patch_w = grid_w / merge_size
        - +1 per row for newline token
        - +2 for begin/end tokens

        Args:
            height: Image height
            width: Image width

        Returns:
            Number of image tokens
        """
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        # Token count: patch_h * (patch_w + 1) + 2
        patch_h = grid_h // self.merge_size
        patch_w = grid_w // self.merge_size

        return patch_h * (patch_w + 1) + 2

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Constructs an image processor from a config dictionary."""
        if "vision_config" not in config_dict:
            config_dict["vision_config"] = {}
        return cls(config=config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Constructs an image processor from a pretrained model."""
        return cls(**kwargs)


class HunYuanVLProcessor(ProcessorMixin):
    """Processor for HunyuanVL that combines image processing and tokenization.

    Handles:
    - Image preprocessing via HunYuanVLImageProcessor
    - Token replacement for image placeholders
    - 4D position_ids construction for xdrope
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # Special token IDs (from HunyuanVL config)
    IMAGE_TOKEN_ID = 120120
    IM_START_TOKEN_ID = 120118
    IM_END_TOKEN_ID = 120119
    PAD_TOKEN_ID = 120002

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = HunYuanVLImageProcessor(**kwargs)

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # Get special token IDs
        self.image_token_id = self.IMAGE_TOKEN_ID
        self.im_start_token_id = self.IM_START_TOKEN_ID
        self.im_end_token_id = self.IM_END_TOKEN_ID
        self.pad_id = self.PAD_TOKEN_ID

        # Get token strings from tokenizer
        if tokenizer is not None:
            self.image_token = tokenizer.convert_ids_to_tokens(self.image_token_id)
            self.im_start_token = tokenizer.convert_ids_to_tokens(
                self.im_start_token_id
            )
            self.im_end_token = tokenizer.convert_ids_to_tokens(self.im_end_token_id)
            self.placeholder_token = tokenizer.convert_ids_to_tokens(
                tokenizer.vocab_size - 1
            )
        else:
            self.image_token = "<image>"
            self.im_start_token = "<im_start>"
            self.im_end_token = "<im_end>"
            self.placeholder_token = "<placeholder>"

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """Process images and text for the model.

        Args:
            images: Single image or list of images
            text: Single text or list of texts
            videos: Video inputs (not currently supported)
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            BatchFeature with:
                - input_ids: Token IDs with image placeholders replaced
                - attention_mask: Attention mask
                - pixel_values: Processed image patches
                - image_grid_thw: Grid dimensions for each image
                - position_ids: 4D position IDs for xdrope
        """
        image_inputs = {}
        videos_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        text = [t for t in text]  # Copy to avoid modifying original

        # Track cumulative image token positions
        image_tokens_cumsum = [0]

        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid_h, grid_w = image_grid_thw[index][-2:]
                    patch_h = grid_h // self.image_processor.merge_size
                    patch_w = grid_w // self.image_processor.merge_size
                    num_image_tokens = patch_h * (patch_w + 1) + 2
                    image_tokens_cumsum.append(
                        image_tokens_cumsum[-1] + num_image_tokens
                    )
                    text[i] = text[i].replace(
                        self.image_token,
                        self.placeholder_token * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace(self.placeholder_token, self.image_token)

        # Pop return_tensors to handle it ourselves at the end
        return_tensors = kwargs.pop("return_tensors", None)

        # Tokenize text
        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)

        # Get input_ids and convert to numpy array for processing
        input_ids = text_inputs["input_ids"]
        if hasattr(input_ids, "tolist"):
            # Handle mlx arrays or torch tensors
            input_ids = np.array(input_ids.tolist())
        elif isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        text_inputs["input_ids"] = input_ids
        seq_len = input_ids.shape[-1]

        # Build 4D position_ids for xdrope
        # Shape: (1, 4, seq_len) where 4 = [base, w, h, t]
        position_ids = np.arange(seq_len)
        position_ids_w = np.arange(seq_len)
        position_ids_h = np.arange(seq_len)
        position_ids_t = np.arange(seq_len)

        if images is not None:
            # Find image token positions
            image_token_pos_indices = np.where(input_ids[0] == self.image_token_id)[0]

            for i in range(len(image_grid_thw)):
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // self.image_processor.merge_size
                patch_w = grid_w // self.image_processor.merge_size

                # Start position for this image's tokens (skip begin token)
                start_pos = image_token_pos_indices[image_tokens_cumsum[i]].item() + 1
                replace_num = (patch_w + 1) * patch_h

                # Set width positions: 0, 1, 2, ..., patch_w, 0, 1, 2, ..., patch_w, ...
                position_ids_w[start_pos : start_pos + replace_num] = np.array(
                    list(range(patch_w + 1)) * patch_h
                )

                # Set height positions: 0, 0, ..., 0, 1, 1, ..., 1, ...
                patch_h_list = []
                for h in range(patch_h):
                    patch_h_list += [h] * (patch_w + 1)
                position_ids_h[start_pos : start_pos + replace_num] = np.array(
                    patch_h_list
                )

                # Set temporal positions: all 0 for images
                position_ids_t[start_pos : start_pos + replace_num] = 0

        # Stack position_ids: (1, 4, seq_len)
        # Order: base, w, h, t
        position_ids = np.stack(
            [position_ids, position_ids_w, position_ids_h, position_ids_t]
        )[np.newaxis, ...]

        text_inputs["position_ids"] = position_ids

        # Build attention mask
        attention_mask = (input_ids != self.pad_id).astype(np.int64)
        text_inputs["attention_mask"] = attention_mask

        # Get image positions
        text_inputs["imgs_pos"] = [self.get_imgs_pos(input_ids[0])]

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        """Apply chat template using the tokenizer."""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def get_imgs_pos(self, doc_ids):
        """Get image positions from document token IDs.

        Args:
            doc_ids: Token IDs array

        Returns:
            List of [start, end] positions for each image
        """
        doc_ids = np.array(doc_ids, dtype=np.int64)
        img_begin_index = np.where(doc_ids == self.im_start_token_id)[0]
        img_end_index = np.where(doc_ids == self.im_end_token_id)[0]
        imgs_pos = np.concatenate(
            (
                np.reshape(img_begin_index + 1, (-1, 1)),
                np.reshape(img_end_index, (-1, 1)),
            ),
            axis=-1,
        ).tolist()
        return imgs_pos

    @property
    def model_input_names(self):
        """Return combined input names from tokenizer and image processor."""
        tokenizer_input_names = (
            self.tokenizer.model_input_names if self.tokenizer else []
        )
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model path."""
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        image_processor = HunYuanVLImageProcessor(**kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)


def split_image_into_patch_blocks(
    pixel_values: np.ndarray,  # shape: [batch_size, 3, H, W]
    patch_size: int = 16,
    adaptor_patch_div: int = 4,
) -> np.ndarray:
    """Split the input image array into large patches and then smaller regions.

    Split the input image tensor (supporting batch) into large patches of size `patch_size`,
    and then further divide each large patch into smaller regions of size
    (patch_size // adaptor_patch_div) x (patch_size // adaptor_patch_div).
    Each small region is extracted as a tensor of shape [3, patch_size, patch_size].
    The final output contains all such small region tensors.

    Args:
        pixel_values: Input image array of shape [batch_size, 3, H, W].
        patch_size: Size of the large patch, e.g., 16.
        adaptor_patch_div: Each large patch is divided into
                          (patch_size // adaptor_patch_div) x (patch_size // adaptor_patch_div)
                          smaller regions.

    Returns:
        patches: An array of shape [N, 3, patch_size, patch_size],
                 where N = batch_size * (H // patch_size) * (W // patch_size) * (patch_size // adaptor_patch_div)^2.
                 Each element in the batch corresponds to one small image region.
    """
    batch_size, channels, height, width = pixel_values.shape
    assert channels == 3, "Pixel values must have 3 channels in dim=1"
    assert (
        height % patch_size == 0 and width % patch_size == 0
    ), "H and W must be divisible by patch_size"

    patch_height_num = height // patch_size
    patch_width_num = width // patch_size

    # Reshape to [B, 3, ph, ps, pw, ps]
    img = pixel_values.reshape(
        batch_size,
        3,
        patch_height_num,
        patch_size,
        patch_width_num,
        patch_size,
    )

    # Further split each psxps patch into (ps//aps)x(ps//aps) small regions
    img = img.reshape(
        batch_size,
        3,
        patch_height_num,
        patch_size // adaptor_patch_div,
        adaptor_patch_div,
        patch_width_num,
        patch_size // adaptor_patch_div,
        adaptor_patch_div,
    )

    # Permute to group the small regions: [B, ph, pw, ps//aps, ps//aps, 3, aps, aps]
    img = img.transpose(0, 2, 5, 3, 6, 1, 4, 7)

    # Reshape into [B * ph * pw * (ps//aps)^2, 3, patch_size, patch_size]
    patches = img.reshape(-1, 3, patch_size, patch_size)

    return patches


# Alias for compatibility
ImageProcessor = HunYuanVLImageProcessor


__all__ = [
    "HunYuanVLImageProcessor",
    "HunYuanVLProcessor",
    "ImageProcessor",
    "smart_resize",
    "split_image_into_patch_blocks",
]
