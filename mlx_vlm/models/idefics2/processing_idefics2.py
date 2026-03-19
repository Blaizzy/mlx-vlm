"""
Processor class for Idefics2.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/processing_idefics2.py
"""

import re
from itertools import accumulate
from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)

from ..base import load_chat_template, to_mlx


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Idefics2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_len"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer=None,
        image_seq_len: int = 64,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        if not hasattr(tokenizer, "image_token"):
            self.fake_image_token = AddedToken(
                "<fake_token_around_image>", normalized=False, special=True
            ).content
            self.image_token = AddedToken(
                "<image>", normalized=False, special=True
            ).content
            tokens_to_add = {
                "additional_special_tokens": [
                    self.fake_image_token,
                    self.image_token,
                ]
            }
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.fake_image_token = tokenizer.image_boundary_token
            self.image_token = tokenizer.image_token
            self.image_token_id = tokenizer.image_token_id

        self.end_of_utterance_token = AddedToken(
            "<end_of_utterance>", normalized=False, special=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.end_of_utterance_token]}
        )
        self.image_seq_len = image_seq_len

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[
            Union[ImageInput, List[ImageInput], List[List[ImageInput]]]
        ] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        return_tensors = kwargs.pop("return_tensors", None)

        n_images_in_text = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )

            fake_image_token = self.fake_image_token
            image_token = self.image_token
            image_str = f"{fake_image_token}{image_token * self.image_seq_len}{fake_image_token}"

            if self.image_processor.do_image_splitting:
                image_str = image_str * 5

            prompt_strings = []
            closing_fake_pattern = re.compile(
                rf"{re.escape(fake_image_token)}(?=[^\s<])"
            )
            for sample in text:
                n_images_in_text.append(sample.count(image_token))
                sample = sample.replace(image_token, image_str)
                sample = sample.replace(
                    f"{fake_image_token}{fake_image_token}",
                    f"{fake_image_token}",
                )
                sample = closing_fake_pattern.sub(f"{fake_image_token} ", sample)
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(prompt_strings, **kwargs)
            inputs.update(text_inputs)

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {image_token} tokens and {len(images)} images."
                        )
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )

            n_images_in_images = [len(sample) for sample in images]
            if text is not None and not n_images_in_images == n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                )

            images = [
                [load_image(im) if is_url(im) else im for im in sample]
                for sample in images
            ]
            image_inputs = self.image_processor(images)
            inputs.update(image_inputs)

        return BatchFeature(data=to_mlx(inputs))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

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

        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("image_seq_len",):
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
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )


__all__ = ["Idefics2Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("idefics2", Idefics2Processor)
