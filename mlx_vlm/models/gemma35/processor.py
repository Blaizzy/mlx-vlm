import inspect
from typing import List, Optional, Union, Dict, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.image_utils import ChannelDimension, PILImageResampling
from transformers.utils import logging

from mlx_vlm.models.base import BaseImageProcessor, expand2square

logger = logging.get_logger(__name__)

# Constants for Gemma35
GEMMA35_MEAN = (0.48145466, 0.4578275, 0.40821073)
GEMMA35_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224  # Default size from vision.py


class Gemma35ImageProcessor(BaseImageProcessor):
    """
    Image processor for Gemma35 model.

    This processor handles image preprocessing for the visual encoder.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Union[int, Dict[str, int], Tuple[int, int]] = IMAGE_SIZE,
        resample=PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size=None,
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = GEMMA35_MEAN,
        image_std: Optional[Union[float, List[float]]] = GEMMA35_STD,
        do_convert_rgb: bool = True,
        do_pad_square: bool = False,
        **kwargs
    ):
        if isinstance(size, (tuple, list)):
            size = {"height": size[0], "width": size[1]}
        elif isinstance(size, int):
            size = {"height": size, "width": size}

        super().__init__(
            image_mean=image_mean,
            image_std=image_std,
            size=(size["height"], size["width"]),
            resample=resample,
            rescale_factor=rescale_factor,
            data_format=ChannelDimension.FIRST
        )

        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.do_pad_square = do_pad_square

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        do_pad_square: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """
        Preprocess an image or a batch of images for use with the Gemma35 model.

        Args:
            images: Image or batch of images to preprocess
            do_resize: Whether to resize the image
            size: Size to resize to
            resample: Resampling method
            do_center_crop: Whether to center crop the image
            crop_size: Size to center crop to
            do_rescale: Whether to rescale pixel values
            rescale_factor: Factor to rescale by
            do_normalize: Whether to normalize pixel values
            image_mean: Mean to use for normalization
            image_std: Standard deviation to use for normalization
            do_convert_rgb: Whether to convert the image to RGB
            do_pad_square: Whether to pad the image to a square
            return_tensors: Type of tensors to return

        Returns:
            Dictionary containing the preprocessed image
        """

        # Set default values based on instance attributes
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pad_square = do_pad_square if do_pad_square is not None else self.do_pad_square
        resample = resample if resample is not None else self.resample

        size = size if size is not None else {"height": self.size[0], "width": self.size[1]}
        if isinstance(size, (list, tuple)):
            size = {"height": size[0], "width": size[1]}
        elif isinstance(size, int):
            size = {"height": size, "width": size}

        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not isinstance(images, list):
            images = [images]

        processed_images = []

        for image in images:
            # Convert to RGB if needed
            if do_convert_rgb and image.mode != "RGB":
                image = image.convert("RGB")

            # Pad to square if needed
            if do_pad_square:
                image = expand2square(image, (0, 0, 0))

            # Resize if needed
            if do_resize:
                image = self.resize(
                    image=image,
                    size=size,
                    resample=resample
                )

            # Center crop if needed
            if do_center_crop:
                crop_size = crop_size if crop_size is not None else self.crop_size
                image = self.center_crop(image=image, crop_size=crop_size)

            # Convert to numpy array
            image_array = np.array(image).astype(np.float32)

            # Rescale if needed
            if do_rescale:
                image_array = self.rescale(image=image_array, scale=rescale_factor)

            # Normalize if needed
            if do_normalize:
                image_array = self.normalize(
                    image=image_array,
                    mean=image_mean,
                    std=image_std
                )

            # Transpose to channels-first format
            if self.data_format == ChannelDimension.FIRST:
                image_array = image_array.transpose(2, 0, 1)

            processed_images.append(image_array)

        # Convert to MLX array
        if return_tensors == "mx" or return_tensors is None:
            pixel_values = mx.array(processed_images)
        else:
            pixel_values = processed_images

        return {"pixel_values": pixel_values}


class Gemma35Processor(ProcessorMixin):
    """
    Processor for Gemma35 model.

    This processor combines the image processor and tokenizer, handling both
    image preprocessing and text tokenization for the Gemma35 model.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs
    ):
        # Initialize image processor
        if image_processor is None:
            image_processor = Gemma35ImageProcessor()
        self.image_processor = image_processor

        # Initialize tokenizer
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True, **kwargs
            )
        self.tokenizer = tokenizer

        self.tokenizer.chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n"


        # Set chat template if provided
        if chat_template is not None:
            if hasattr(self.tokenizer, "chat_template"):
                self.tokenizer.chat_template = chat_template

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[Image.Image, List[Image.Image]] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "mx",
        **kwargs
    ) -> Dict[str, mx.array]:
        """
        Process inputs for the Gemma35 model.

        Args:
            text: Text input or list of text inputs
            images: Image input or list of image inputs
            padding: Padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Type of tensors to return
            **kwargs: Additional arguments

        Returns:
            Dictionary of processed inputs
        """
        processed_inputs = {}

        # Process images if provided
        if images is not None:
            image_features = self.image_processor.preprocess(
                images=images,
                return_tensors=return_tensors,
                **kwargs
            )
            processed_inputs.update(image_features)

        # Process text if provided
        if text is not None:
            self.tokenizer.padding_side = "right"  # Default padding side
            text_inputs = self.tokenizer(
                text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs
            )
            processed_inputs.update(text_inputs)

        return processed_inputs

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode method"""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode method"""
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """Save the processor to the specified directory"""
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        # No need to explicitly save the image processor as it doesn't have pretrained weights

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model"""
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        image_processor = Gemma35ImageProcessor(**kwargs)
        return Gemma35Processor(
            image_processor=image_processor, tokenizer=tokenizer
        )

# Register the processor with AutoProcessor
MODEL_TYPE = "gemma35"
AutoProcessor.register(MODEL_TYPE, Gemma35Processor)

logger.info(f"Registered Gemma35Processor class for model type '{MODEL_TYPE}'.")
