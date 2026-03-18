"""
Processor class for Llama4.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/processing_llama4.py
"""

from typing import List, Optional, Union

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import to_mlx


class Llama4Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        pixel_shuffle_ratio: float = 0.5,
        fake_image_token="<|image|>",
        image_token="<|image|>",
        start_of_image_token="<|image_start|>",
        end_of_image_token="<|image_end|>",
        patch_token="<|patch|>",
        tile_x_separator_token="<|tile_x_separator|>",
        tile_y_separator_token="<|tile_y_separator|>",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        self.patch_size = patch_size

        self.fake_image_token = fake_image_token
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.start_of_img_token = start_of_image_token
        self.end_of_img_token = end_of_image_token
        self.img_patch_token = patch_token
        self.tile_token = tile_x_separator_token
        self.tile_global_token = tile_y_separator_token

    def _prompt_split_image(self, aspect_ratio, num_patches_per_chunk):
        """Create a structured string representation of image tokens."""
        img_string = "<|image_start|>"
        ratio_h, ratio_w = aspect_ratio
        if ratio_h * ratio_w > 1:
            for yy in range(ratio_h):
                for xx in range(ratio_w):
                    img_string += "<|patch|>" * num_patches_per_chunk
                    if xx < ratio_w - 1:
                        img_string += "<|tile_x_separator|>"
                img_string += "<|tile_y_separator|>"

        img_string += "<|image|>"
        img_string += "<|patch|>" * num_patches_per_chunk
        img_string += "<|image_end|>"
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
            image_inputs = self.image_processor(images=images)
            image_height, image_width = image_inputs["pixel_values"][0].shape[-2:]
            num_patches_per_chunk = int(
                (image_height // self.patch_size)
                * (image_width // self.patch_size)
                // self.downsample_ratio
            )
            aspect_ratios = image_inputs.pop("aspect_ratios")

            total_placeholders = sum(
                prompt.count(self.fake_image_token) for prompt in text
            )
            if total_placeholders != len(images):
                raise ValueError(
                    f"Found {total_placeholders} placeholders across the batch, "
                    f"but have {len(images)} flattened images."
                )

            image_index = 0
            processed_text = []
            for prompt in text:
                placeholder_count = prompt.count(self.fake_image_token)
                if placeholder_count == 0:
                    processed_text.append(prompt)
                    continue
                prompt_splits = prompt.split(self.fake_image_token)
                new_prompt = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_prompt.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = self._prompt_split_image(
                            aspect_ratios[image_index],
                            num_patches_per_chunk,
                        )
                        image_index += 1
                        new_prompt.append(tokens_for_this_image)
                processed_text.append("".join(new_prompt))

            if image_index != len(images):
                raise ValueError(
                    "Number of image placeholders in the prompt does not match the number of images."
                )

            text = processed_text

        return_tensors = kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **kwargs)

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


__all__ = ["Llama4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("llama4", Llama4Processor)
