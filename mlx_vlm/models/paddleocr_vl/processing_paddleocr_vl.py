import math
from typing import List, Optional, Union

import numpy as np
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
from transformers.utils import logging

logger = logging.get_logger(__name__)


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
):
    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class ImageProcessor(BaseImageProcessor):
    """
    MLX-native image processor for PaddleOCRVL that doesn't require torch.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 147384,
        max_pixels: int = 2822400,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError(
                    "size must contain 'shortest_edge' and 'longest_edge' keys."
                )
        else:
            size = {"shortest_edge": 147384, "longest_edge": 2822400}
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError(
                    "size must contain 'shortest_edge' and 'longest_edge' keys."
                )
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = (
            temporal_patch_size
            if temporal_patch_size is not None
            else self.temporal_patch_size
        )
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        if images is not None:
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        data = {}
        pixel_values, vision_grid_thws = [], []
        if images is not None:
            processed_images = []
            for image in images:
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = image.resize((resized_width, resized_height), resample)
                img_array = to_numpy_array(image)

                if do_rescale:
                    img_array = img_array / 255.0

                if do_normalize:
                    mean = np.array(self.image_mean).reshape(1, 1, 3)
                    std = np.array(self.image_std).reshape(1, 1, 3)
                    img_array = (img_array - mean) / std

                processed_images.append(img_array)

            patches = np.array(processed_images)

            if patches.shape[1] > 3:
                patches = patches.transpose(0, 3, 1, 2)
            if patches.shape[0] == 1:
                patches = np.tile(patches, (temporal_patch_size, 1, 1, 1))

            channel = patches.shape[1]
            grid_t = patches.shape[0] // temporal_patch_size
            grid_h, grid_w = (
                resized_height // patch_size,
                resized_width // patch_size,
            )
            patches = patches.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
            if temporal_patch_size != 1:
                raise ValueError(
                    f"temporal_patch_size must be 1!, but got {temporal_patch_size}!"
                )
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel, patch_size, patch_size
            )
            image_grid_thw = (grid_t, grid_h, grid_w)
            pixel_values.extend(flatten_patches)
            vision_grid_thws.append(image_grid_thw)

        pixel_values = np.array([pixel_values])
        vision_grid_thws = np.array(vision_grid_thws)
        data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        return BatchFeature(data, tensor_type=return_tensors)


class PaddleOCRVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = ImageProcessor(**kwargs)

        self.tokenizer = tokenizer
        self.image_token = (
            "<|IMAGE_PLACEHOLDER|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.image_processor = image_processor

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
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

        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        text = [t for t in text]  # Copy to avoid modifying original

        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>"
                        * (
                            image_grid_thw[index].prod()
                            // self.image_processor.merge_size
                            // self.image_processor.merge_size
                        ),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

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

        return BatchFeature(
            data={**text_inputs, **image_inputs},
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
        image_processor = ImageProcessor(**kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)


__all__ = [
    "PaddleOCRVLProcessor",
    "ImageProcessor",
]
