from transformers import Qwen2_5_VLProcessor

from ..base import install_auto_processor_patch


class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.image_token = (
            "<|imgpad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.image_token_id = (
            151665
            if not hasattr(tokenizer, "image_token_id")
            else tokenizer.image_token_id
        )


install_auto_processor_patch("dots_ocr", DotsVLProcessor)

__all__ = ["DotsVLProcessor"]
