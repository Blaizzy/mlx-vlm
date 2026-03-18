"""
Processor class for DeepSeek-VL (multi_modality).

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_vl/processing_deepseek_vl.py
"""

from typing import List, Optional, Union

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import to_mlx


class MultiModalityProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "num_image_tokens"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        num_image_tokens=576,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.num_image_tokens = num_image_tokens

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        images: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (
                isinstance(text, (list, tuple))
                and all(isinstance(t, str) for t in text)
            ):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )

        return_tensors = kwargs.pop("return_tensors", None)

        prompt_strings = []
        one_img_tokens = self.image_token * self.num_image_tokens
        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **kwargs)

        if images is not None:
            data["pixel_values"] = self.image_processor(images)["pixel_values"]

        return BatchFeature(data=to_mlx(data))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["MultiModalityProcessor"]
