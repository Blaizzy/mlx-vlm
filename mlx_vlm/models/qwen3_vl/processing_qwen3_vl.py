"""
Processor class for Qwen3VL.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


class Qwen3VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

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

        self.vision_start_token = (
            "<|vision_start|>"
            if not hasattr(tokenizer, "vision_start_token")
            else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>"
            if not hasattr(tokenizer, "vision_end_token")
            else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
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
        else:
            image_grid_thw = None

        if videos is not None:
            _video_proc = self.video_processor or self.image_processor
            videos_inputs = _video_proc(videos=videos)
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
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

        if video_grid_thw is not None:
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
        import json
        from pathlib import Path

        from transformers import AutoImageProcessor, AutoTokenizer

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Read processor_config.json for correct init kwargs
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("chat_template",):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]
            ip_cfg = proc_cfg.get("image_processor", {})
            if "patch_size" in ip_cfg:
                ip_overrides["patch_size"] = ip_cfg["patch_size"]
            if "size" in ip_cfg:
                ip_overrides["size"] = ip_cfg["size"]

        try:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=False,
                **ip_overrides,
                **kwargs,
            )
        except ValueError:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                **ip_overrides,
                **kwargs,
            )

        video_processor = None
        try:
            from transformers import AutoVideoProcessor

            video_processor = AutoVideoProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        except (ImportError, ValueError, OSError):
            pass

        if "chat_template" not in proc_kwargs:
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template is not None:
                proc_kwargs["chat_template"] = chat_template

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            **proc_kwargs,  # may include chat_template from processor_config.json
        )


__all__ = ["Qwen3VLProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("qwen3_vl", Qwen3VLProcessor)
