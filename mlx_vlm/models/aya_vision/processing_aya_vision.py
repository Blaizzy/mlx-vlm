"""
Processor class for AyaVision.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/aya_vision/processing_aya_vision.py
"""

from typing import List, Optional, Union

import numpy as np
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import to_mlx


class AyaVisionProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "img_size",
        "image_token",
        "downsample_factor",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 28,
        img_size: int = 364,
        image_token="<image>",
        downsample_factor: int = 1,
        start_of_img_token="<|START_OF_IMG|>",
        end_of_img_token="<|END_OF_IMG|>",
        img_patch_token="<|IMG_PATCH|>",
        img_line_break_token="<|IMG_LINE_BREAK|>",
        tile_token="TILE",
        tile_global_token="TILE_GLOBAL",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token = image_token
        self.patch_size = patch_size * downsample_factor
        self.img_size = img_size

        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token
        self.tile_token = tile_token
        self.tile_global_token = tile_global_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.img_patch_token)
        self.image_ids = tokenizer.convert_tokens_to_ids(
            [
                img_patch_token,
                tile_token,
                tile_global_token,
                start_of_img_token,
                end_of_img_token,
            ]
        )

    def _prompt_split_image(self, num_patches):
        """Create a structured string representation of image tokens."""
        img_patches_per_tile = (self.img_size // self.patch_size) ** 2
        img_string = f"{self.start_of_img_token}"
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += (
                    f"{self.tile_token}_{idx}"
                    + f"{self.img_patch_token}" * img_patches_per_tile
                )

        img_string += (
            f"{self.tile_global_token}"
            + f"{self.img_patch_token}" * img_patches_per_tile
        )
        img_string += f"{self.end_of_img_token}"
        return img_string

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images
        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)

            # Separate image kwargs
            images_kwargs = {}
            for k in list(kwargs.keys()):
                if k in ("crop_to_patches",):
                    images_kwargs[k] = kwargs.pop(k)

            image_inputs = self.image_processor(images=images, **images_kwargs)
            num_patches = image_inputs.pop("num_patches")
            image_index = 0
            processed_text = []
            for prompt in text:
                new_prompt = prompt
                while "<image>" in new_prompt:
                    image_tokens = self._prompt_split_image(num_patches[image_index])
                    new_prompt = new_prompt.replace("<image>", image_tokens, 1)
                    image_index += 1
                processed_text.append(new_prompt)

            if image_index != len(images):
                raise ValueError(
                    "Number of image placeholders in the prompt does not match the number of images."
                )

            text = processed_text

        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **kwargs, return_tensors=None)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["AyaVisionProcessor"]
