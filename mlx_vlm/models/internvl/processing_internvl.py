import json
from pathlib import Path

from transformers import AutoTokenizer

from ..base import install_auto_processor_patch, load_chat_template
from ..internvl_chat.processor import (
    IMG_CONTEXT_TOKEN,
    InternVLChatProcessor,
    InternVLImageProcessor,
)
from .config import VisionConfig


class InternVLProcessor(InternVLChatProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        if isinstance(text, str):
            text = [text]
        if images is None:
            return self.tokenizer(text, **kwargs)
        text = [prompt.replace(IMG_CONTEXT_TOKEN, "<image>") for prompt in text]
        inputs = super().__call__(text=text, images=images, **kwargs)
        inputs.pop("num_patches_list", None)
        return inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_path = Path(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        load_chat_template(tokenizer, model_path)
        config = json.loads((model_path / "config.json").read_text())
        processor_config = json.loads(
            (model_path / "processor_config.json").read_text()
        )
        vision_config = VisionConfig.from_dict(config["vision_config"])
        image_processor = InternVLImageProcessor(
            size=vision_config.image_size,
            dynamic_max_num=config.get("max_dynamic_patch", 12),
            dynamic_min_num=config.get("min_dynamic_patch", 1),
            dynamic_use_thumbnail=config.get("use_thumbnail", True),
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            num_image_token=processor_config.get("image_seq_length", 256),
            chat_template=tokenizer.chat_template,
        )


install_auto_processor_patch("internvl", InternVLProcessor)
