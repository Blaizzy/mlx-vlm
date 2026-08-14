import os
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers import BatchFeature

from ..base import install_auto_processor_patch

GOT_OCR_MEAN = (0.48145466, 0.4578275, 0.40821073)
GOT_OCR_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_TOKEN_LEN = 256

# The MPT conversation GOT was trained with, from modeling_GOT.py. The tokenizer
# ships no chat template, so otherwise the model gets a bare instruction.
GOT_SYSTEM = (
    "<|im_start|>system\n"
    "        You should follow the instructions carefully and "
    "explain your answers in detail."
)
GOT_SEP = "<|im_end|>"
GOT_USER_ROLE = "<|im_start|>user\n"
GOT_ASSISTANT_ROLE = "<|im_start|>assistant\n"


def build_got_prompt(instruction: str, has_image: bool = True) -> str:
    """Wrap an OCR instruction in the MPT conversation GOT expects."""
    if GOT_SEP in instruction or GOT_USER_ROLE in instruction:
        return instruction

    image_tokens = "<img>" + "<imgpad>" * IMAGE_TOKEN_LEN + "</img>\n"
    if has_image:
        if "<image>" in instruction:
            instruction = instruction.replace("<image>", image_tokens)
        else:
            instruction = image_tokens + instruction

    return (
        GOT_SYSTEM
        + GOT_SEP
        + GOT_USER_ROLE
        + instruction
        + GOT_SEP
        + GOT_ASSISTANT_ROLE
    )


class GotOcrImageProcessor:
    """Image processor for GOT-OCR."""

    def __init__(
        self,
        image_size=1024,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        self.image_size = int(image_size)
        self.do_normalize = bool(do_normalize)
        self.image_mean = list(image_mean or GOT_OCR_MEAN)
        self.image_std = list(image_std or GOT_OCR_STD)

    def preprocess(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        for image in images:
            image = image.convert("RGB")
            # Resize using BICUBIC exactly to image_size x image_size
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

            arr = np.asarray(image)

            # ToTensor: [0, 1]
            arr = arr.astype(np.float32) / 255.0
            pixel_values.append(arr)

        pixel_values = np.stack(pixel_values)

        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array(self.image_std, dtype=np.float32).reshape(1, 1, 1, 3)
            pixel_values = (pixel_values - mean) / std

        return {"pixel_values": pixel_values}

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)


class GotOcrProcessor:
    """Processor for GOT-OCR."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "GotOcrImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            image_processor = GotOcrImageProcessor(**kwargs)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        text: Union[str, List[str]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        has_image = images is not None
        if has_image:
            image_inputs = self.image_processor(images, **kwargs)
        else:
            image_inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = build_got_prompt(text, has_image=has_image)
            elif isinstance(text, list):
                text = [build_got_prompt(t, has_image=has_image) for t in text]

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            text_inputs = {}

        return BatchFeature(data={**image_inputs, **text_inputs})

    def save_pretrained(self, save_directory, **kwargs):
        # Writes qwen.tiktoken, which the tokenizer needs to reload. Without
        # this, convert.py copies only *.py and *.json and the converted repo
        # ships no vocab file, hidden locally by the HF cache still having one.
        os.makedirs(save_directory, exist_ok=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        trust_remote_code = True
        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)

        image_processor = GotOcrImageProcessor()
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)


install_auto_processor_patch("got", GotOcrProcessor)
