"""Custom processor for FastVLM - MLX-native implementation without torch/timm dependency."""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch

# Special tokens
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def expand_to_square(image: np.ndarray, background_color: float = 0.0) -> np.ndarray:
    """
    Expand an image to a square by padding with background color.

    Args:
        image: Image array in HWC format with values in [0, 1]
        background_color: Value to fill padding with (default 0.0)

    Returns:
        Square image with padding added to shorter dimension
    """
    height, width = image.shape[:2]
    if width == height:
        return image

    size = max(height, width)
    channels = image.shape[2] if len(image.shape) == 3 else 1

    if len(image.shape) == 3:
        result = np.full((size, size, channels), background_color, dtype=image.dtype)
    else:
        result = np.full((size, size), background_color, dtype=image.dtype)

    if width > height:
        # Pad top and bottom
        offset = (width - height) // 2
        result[offset : offset + height, :] = image
    else:
        # Pad left and right
        offset = (height - width) // 2
        result[:, offset : offset + width] = image

    return result


def resize_image(
    image: np.ndarray,
    size: int,
    resample: PILImageResampling = PILImageResampling.BICUBIC,
) -> np.ndarray:
    """
    Resize image to target size using PIL.

    Args:
        image: Image array in HWC format with values in [0, 1]
        size: Target size (square)
        resample: Resampling method

    Returns:
        Resized image
    """
    # Convert to uint8 for PIL
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    pil_image = Image.fromarray(image_uint8)

    # Map resampling method
    resample_map = {
        PILImageResampling.NEAREST: Image.Resampling.NEAREST,
        PILImageResampling.BILINEAR: Image.Resampling.BILINEAR,
        PILImageResampling.BICUBIC: Image.Resampling.BICUBIC,
        PILImageResampling.LANCZOS: Image.Resampling.LANCZOS,
        PILImageResampling.BOX: Image.Resampling.BOX,
        PILImageResampling.HAMMING: Image.Resampling.HAMMING,
    }
    pil_resample = resample_map.get(resample, Image.Resampling.BICUBIC)

    # Resize
    resized = pil_image.resize((size, size), resample=pil_resample)

    # Convert back to float32 [0, 1]
    return np.array(resized, dtype=np.float32) / 255.0


def normalize_image(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """
    Normalize image with mean and std.

    Args:
        image: Image array in HWC format
        mean: Mean values per channel
        std: Std values per channel

    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    return (image - mean) / std


class FastVLMImageProcessor(BaseImageProcessor):
    """
    MLX-native image processor for FastVLM.

    Handles:
    - Expand to square (padding shorter dimension)
    - Resize to target size (default 1024x1024)
    - Normalize with mean/std
    """

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        size: Optional[dict] = None,
        crop_size: Optional[dict] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        rescale_factor: float = 1 / 255,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Default to 1024 for FastVLM
        size = size if size is not None else {"shortest_edge": 1024}
        crop_size = (
            crop_size if crop_size is not None else {"height": 1024, "width": 1024}
        )

        self.size = size
        self.crop_size = crop_size
        self.resample = resample
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.rescale_factor = rescale_factor

        # FastVLM uses mean=0, std=1 (just rescale to [0, 1])
        self.image_mean = image_mean if image_mean is not None else [0.0, 0.0, 0.0]
        self.image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]

    def preprocess(
        self,
        images: ImageInput,
        size: Optional[dict] = None,
        crop_size: Optional[dict] = None,
        resample: Optional[PILImageResampling] = None,
        do_resize: Optional[bool] = None,
        do_center_crop: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess images for FastVLM."""
        # Use instance defaults if not provided
        size = size if size is not None else self.size
        crop_size = crop_size if crop_size is not None else self.crop_size
        resample = resample if resample is not None else self.resample
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_center_crop = (
            do_center_crop if do_center_crop is not None else self.do_center_crop
        )
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        # Get target size
        if "shortest_edge" in size:
            target_size = size["shortest_edge"]
        elif "height" in size:
            target_size = size["height"]
        else:
            target_size = 1024

        # Handle crop size
        if "height" in crop_size:
            crop_h, crop_w = crop_size["height"], crop_size["width"]
        else:
            crop_h = crop_w = target_size

        # Normalize images input
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be PIL.Image.Image, numpy.ndarray, etc."
            )

        # Convert to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to numpy
        images = [to_numpy_array(image) for image in images]

        # Store original sizes before processing
        image_sizes = [
            (img.shape[1], img.shape[0]) for img in images
        ]  # (width, height)

        processed_images = []
        for image in images:
            # Rescale to [0, 1] if needed
            if do_rescale and image.max() > 1.0:
                image = image.astype(np.float32) * self.rescale_factor

            # Expand to square
            image = expand_to_square(image, background_color=0.0)

            # Resize to target size
            if do_resize:
                image = resize_image(image, target_size, resample)

            # Center crop (if different from resize)
            if do_center_crop and (crop_h != target_size or crop_w != target_size):
                h, w = image.shape[:2]
                top = (h - crop_h) // 2
                left = (w - crop_w) // 2
                image = image[top : top + crop_h, left : left + crop_w]

            # Normalize
            if do_normalize:
                image = normalize_image(image, image_mean, image_std)

            processed_images.append(image)

        # Stack and convert to NCHW format
        pixel_values = np.stack(processed_images, axis=0)
        pixel_values = pixel_values.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            },
            tensor_type=return_tensors,
        )


class FastVLMProcessor(ProcessorMixin):
    """
    Processor for FastVLM that combines image processor and tokenizer.

    Handles:
    - Image preprocessing via FastVLMImageProcessor
    - Token replacement for image placeholder
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "FastVLMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = FastVLMImageProcessor()

        self.image_token = DEFAULT_IMAGE_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Optional[Union[str, List[str]]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process images and text for FastVLM.

        Args:
            images: Single image or list of images
            text: Single text or list of texts
            return_tensors: Return tensor type (None, "np", "pt")

        Returns:
            BatchFeature with input_ids, attention_mask, pixel_values
        """
        # Normalize text input to List[str]
        if text is None:
            text = [""]
        elif isinstance(text, str):
            text = [text]
        # else: text is already a list

        # Process images
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=None)

        input_ids_list = []
        attention_mask_list = []

        for prompt in text:
            # Find image token positions
            parts = prompt.split(self.image_token)

            if len(parts) == 1:
                # No image token, just tokenize
                tokens = self.tokenizer(
                    prompt, return_tensors=None, add_special_tokens=False
                )
                sample_ids = np.array(tokens["input_ids"], dtype=np.int64)
            else:
                # Has image token(s) - tokenize parts separately and insert image token
                all_ids = []
                for i, part in enumerate(parts):
                    if part:  # Skip empty strings
                        part_tokens = self.tokenizer(
                            part, return_tensors=None, add_special_tokens=False
                        )
                        all_ids.append(
                            np.array(part_tokens["input_ids"], dtype=np.int64)
                        )

                    # Add image token between parts (not after last part)
                    if i < len(parts) - 1:
                        all_ids.append(
                            np.array([self.image_token_index], dtype=np.int64)
                        )

                sample_ids = (
                    np.concatenate(all_ids) if all_ids else np.array([], dtype=np.int64)
                )

            # Add batch dimension
            if sample_ids.ndim == 1:
                sample_ids = sample_ids[np.newaxis, :]

            sample_mask = np.ones_like(sample_ids)

            input_ids_list.append(sample_ids)
            attention_mask_list.append(sample_mask)

        # Stack if all same length, otherwise keep as list
        if len(input_ids_list) == 1:
            input_ids = input_ids_list[0]
            attention_mask = attention_mask_list[0]
        else:
            # Pad to same length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - ids.shape[1]
                if pad_len > 0:
                    ids = np.pad(ids, ((0, 0), (0, pad_len)), constant_values=0)
                    mask = np.pad(mask, ((0, 0), (0, pad_len)), constant_values=0)
                padded_ids.append(ids)
                padded_masks.append(mask)
            input_ids = np.concatenate(padded_ids, axis=0)
            attention_mask = np.concatenate(padded_masks, axis=0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **image_inputs,
        }

        return BatchFeature(data=result, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        """Apply chat template using the tokenizer."""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        """Return combined input names from tokenizer and image processor."""
        tokenizer_names = self.tokenizer.model_input_names if self.tokenizer else []
        image_names = (
            self.image_processor.model_input_names
            if hasattr(self.image_processor, "model_input_names")
            else []
        )
        return list(dict.fromkeys(tokenizer_names + image_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model path."""
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Load image processor config
        image_processor_config = {}
        try:
            if is_local:
                config_path = model_path / "preprocessor_config.json"
            else:
                config_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "preprocessor_config.json"
                    )
                )

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Extract relevant config keys
                for key in [
                    "size",
                    "crop_size",
                    "resample",
                    "do_resize",
                    "do_center_crop",
                    "do_rescale",
                    "do_normalize",
                    "do_convert_rgb",
                    "rescale_factor",
                    "image_mean",
                    "image_std",
                ]:
                    if key in config:
                        image_processor_config[key] = config[key]
        except Exception:
            pass

        image_processor = FastVLMImageProcessor(**image_processor_config)

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **kwargs,
        )


# Patch AutoProcessor for FastVLM models (fastvlm and llava_qwen2 model types)
install_auto_processor_patch(["fastvlm", "llava_qwen2"], FastVLMProcessor)


__all__ = ["FastVLMImageProcessor", "FastVLMProcessor"]
