import json
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from transformers import AutoTokenizer, BatchFeature, ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template
from ..internvl_chat.processor import InternVLImageProcessor


class InternVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length=256,
        chat_template=None,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_seq_length = image_seq_length
        self.image_token = "<IMG_CONTEXT>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Optional[List[Image.Image]] = None,
        return_tensors="mlx",
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        data = {}
        patch_counts = []
        if images is not None:
            image_inputs = self.image_processor.preprocess(
                images, return_tensors=return_tensors
            )
            patch_counts = image_inputs.pop("num_patches_list")
            data.update(image_inputs)

        image_index = 0
        processed_text = []
        for prompt in text:
            while self.image_token in prompt and image_index < len(patch_counts):
                tokens = (
                    self.image_token * self.image_seq_length * patch_counts[image_index]
                )
                prompt = prompt.replace(self.image_token, f"<img>{tokens}</img>", 1)
                image_index += 1
            processed_text.append(prompt)
        if image_index != len(patch_counts):
            raise ValueError(
                "Number of image placeholders does not match the number of images."
            )

        text_inputs = self.tokenizer(
            processed_text, return_tensors=return_tensors, **kwargs
        )
        return BatchFeature(data={**text_inputs, **data})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_path = Path(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        load_chat_template(tokenizer, model_path)
        config = json.loads((model_path / "config.json").read_text())
        processor_config = json.loads(
            (model_path / "processor_config.json").read_text()
        )
        vision_config = config["vision_config"]
        image_size = vision_config.get("image_size", 448)
        patch_size = vision_config.get("patch_size", 14)
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        image_processor = InternVLImageProcessor(
            size=image_size,
            dynamic_max_num=config.get("max_dynamic_patch", 12),
            dynamic_min_num=config.get("min_dynamic_patch", 1),
            dynamic_use_thumbnail=config.get("use_thumbnail", True),
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=processor_config.get("image_seq_length", 256),
            chat_template=tokenizer.chat_template,
        )


install_auto_processor_patch("internvl", InternVLProcessor)
