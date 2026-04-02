from transformers import Qwen2_5_VLProcessor
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template


class DotsDummyVideoProcessor:
    merge_size = 2
    temporal_patch_size = 1
    model_input_names = []

    def __call__(self, videos=None, **kwargs):
        raise NotImplementedError("DOTS MLX processors do not support video inputs.")


class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        video_processor = kwargs.pop("video_processor", DotsDummyVideoProcessor())
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        ProcessorMixin.__init__(
            self,
            image_processor,
            tokenizer,
            chat_template=chat_template or getattr(tokenizer, "chat_template", None),
        )
        self.video_processor = video_processor
        self.image_token = (
            "<|imgpad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.image_token_id = 151665
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.video_token_id = 151656

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from pathlib import Path

        from transformers import AutoImageProcessor, AutoTokenizer

        kwargs.pop("use_fast", None)
        model_path = Path(pretrained_model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if model_path.exists():
            load_chat_template(tokenizer, model_path)

        try:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=False,
                **kwargs,
            )
        except ValueError:
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


install_auto_processor_patch("dots_ocr", DotsVLProcessor)

__all__ = ["DotsVLProcessor"]
