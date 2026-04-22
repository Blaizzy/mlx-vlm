"""
Processor class for Qwen2.5VL.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import to_mlx


class Qwen2_5_VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    # Override the check_argument_for_proper_class method to allow for numpy processors
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

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
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        videos_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            _video_proc = self.video_processor or self.image_processor
            videos_inputs = _video_proc(videos=videos)
            video_grid_thw = videos_inputs["video_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            _video_proc = self.video_processor or self.image_processor
            merge_length = _video_proc.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * num_video_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **kwargs)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **videos_inputs})
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + image_processor_input_names
                + ["mm_token_type_ids"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        from ..base import load_chat_template
        from ..qwen3_vl.processing_qwen3_vl import (
            Qwen3VLImageProcessor,
            Qwen3VLVideoProcessor,
            _load_qwen_vl_json,
            _qwen_vl_image_kwargs,
            _qwen_vl_video_kwargs,
        )

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Qwen2.5 default: patch_size=14.
        image_processor = Qwen3VLImageProcessor(
            **_qwen_vl_image_kwargs(
                pretrained_model_name_or_path, default_patch_size=14
            )
        )
        video_processor = Qwen3VLVideoProcessor(
            **_qwen_vl_video_kwargs(
                pretrained_model_name_or_path, default_patch_size=14
            )
        )

        proc_cfg = (
            _load_qwen_vl_json(pretrained_model_name_or_path, "processor_config.json")
            or {}
        )
        chat_template = proc_cfg.get(
            "chat_template", getattr(tokenizer, "chat_template", None)
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )


__all__ = ["Qwen2_5_VLProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("qwen2_5_vl", Qwen2_5_VLProcessor)
