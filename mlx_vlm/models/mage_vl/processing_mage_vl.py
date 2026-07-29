"""mlx-vlm processor for Mage-VL (image path).

Reuses HuggingFace's *slow* ``Qwen2VLImageProcessor`` (numpy, no torch) for
pixel preprocessing, expands ``<|image_pad|>`` placeholders to the merged token
count, and returns mlx arrays. Registered on ``AutoProcessor`` for the
``mage_vl`` model type so ``mlx_vlm`` picks it up without ``trust_remote_code``.

Video (the DCVC neural codec) is out of scope here — image inference only.
The model recomputes vision ``patch_positions`` internally from ``image_grid_thw``,
so the processor only needs to emit ``input_ids`` / ``pixel_values`` /
``image_grid_thw``.
"""
from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx


class MageVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # torch-free envs: skip HF's isinstance validation (our numpy processors are
    # duck-typed to the call sites).
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, videos=None, **kwargs) -> BatchFeature:
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="np")
            image_grid_thw = image_inputs["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]
        text = list(text)

        if image_grid_thw is not None:
            merge = self.image_processor.merge_size**2
            idx = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n = int(np.prod(image_grid_thw[idx]) // merge)
                    text[i] = text[i].replace(self.image_token, "<|ph|>" * n, 1)
                    idx += 1
                text[i] = text[i].replace("<|ph|>", self.image_token)

        kwargs.pop("return_tensors", None)
        kwargs.pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **kwargs)
        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer, Qwen2VLImageProcessor

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("_from_auto", None)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        # SLOW image processor: the Fast variant changes pixel_values bit-for-bit.
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


__all__ = ["MageVLProcessor"]

install_auto_processor_patch("mage_vl", MageVLProcessor)
