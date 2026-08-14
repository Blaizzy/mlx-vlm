from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers import BatchFeature

from ..base import install_auto_processor_patch

GOT_OCR_MEAN = (0.48145466, 0.4578275, 0.40821073)
GOT_OCR_STD = (0.26862954, 0.26130258, 0.27577711)


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
        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
            # Insert image tokens if images are provided
            image_tokens = "<img>" + "<imgpad>" * 256 + "</img>\n"
            if text is not None:
                if isinstance(text, str):
                    if "<image>" in text:
                        text = text.replace("<image>", image_tokens)
                    else:
                        text = image_tokens + text
                elif isinstance(text, list):
                    new_text = []
                    for t in text:
                        if "<image>" in t:
                            new_text.append(t.replace("<image>", image_tokens))
                        else:
                            new_text.append(image_tokens + t)
                    text = new_text
        else:
            image_inputs = {}

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            text_inputs = {}

        return BatchFeature(data={**image_inputs, **text_inputs})

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
