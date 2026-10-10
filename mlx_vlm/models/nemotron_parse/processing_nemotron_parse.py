from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers import BatchFeature

from ..base import install_auto_processor_patch

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _resize_with_aspect_ratio(image: np.ndarray, max_size) -> np.ndarray:
    """Resize image maintaining aspect ratio with a (height, width) limit."""
    max_size_height, max_size_width = max_size
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_height = height
    new_width = width

    if height > max_size_height:
        new_height = max_size_height
        new_width = int(new_height * aspect_ratio)

    if new_width > max_size_width:
        new_width = max_size_width
        new_height = int(new_width / aspect_ratio)

    if new_height == height and new_width == width:
        return image

    resized = Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR)
    return np.asarray(resized)


def _pad_to_size(image: np.ndarray, target_size) -> np.ndarray:
    """Center-pad image to target size with white padding."""
    target_height, target_width = target_size
    h, w = image.shape[:2]
    pad_h = max(0, target_height - h)
    pad_w = max(0, target_width - w)

    if pad_h == 0 and pad_w == 0:
        return image

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(image.shape) == 3:
        return np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255,
        )
    return np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=255,
    )


class NemotronParseImageProcessor:
    """Image processor for Nemotron-Parse."""

    def __init__(
        self,
        final_size=(2048, 1664),
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        if isinstance(final_size, (list, tuple)) and len(final_size) >= 2:
            self.final_size = (int(final_size[0]), int(final_size[1]))
        else:
            self.final_size = (2048, 1664)

        self.do_normalize = bool(do_normalize)
        self.image_mean = list(image_mean or OPENAI_CLIP_MEAN)
        self.image_std = list(image_std or OPENAI_CLIP_STD)

    def preprocess(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        for image in images:
            image = image.convert("RGB")
            arr = np.asarray(image)
            arr = _resize_with_aspect_ratio(arr, self.final_size)
            arr = _pad_to_size(arr, self.final_size)

            # ToTensor: [0, 1] and channels-first.
            arr = arr.astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            pixel_values.append(arr)

        pixel_values = np.stack(pixel_values)

        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array(self.image_std, dtype=np.float32).reshape(1, 3, 1, 1)
            pixel_values = (pixel_values - mean) / std

        return {"pixel_values": pixel_values}

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)


class NemotronParseProcessor:
    """Processor for Nemotron-Parse."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "NemotronParseImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            image_processor = NemotronParseImageProcessor(**kwargs)
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

        trust_remote_code = kwargs.get("trust_remote_code", None)
        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)

        image_processor = NemotronParseImageProcessor()
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)


install_auto_processor_patch("nemotron_parse", NemotronParseProcessor)
