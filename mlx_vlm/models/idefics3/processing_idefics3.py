"""
Processor class for Idefics3.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py
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


def _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len, fake_token_around_image, image_token, global_img_token
):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_img_token,
    )


class Idefics3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_len"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer=None,
        image_seq_len: int = 169,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.fake_image_token = AddedToken(
            "<fake_token_around_image>", normalized=False, special=True
        ).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.end_of_utterance_token = AddedToken(
            "<end_of_utterance>", normalized=False, special=True
        ).content
        self.global_image_tag = "<global-img>"
        self.image_seq_len = image_seq_len
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(
            self.fake_image_token
        )
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(
            self.global_image_tag
        )

        self._regex_to_remove_extra_special_tokens = re.compile(
            r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+"
        )

        tokens_to_add = {
            "additional_special_tokens": [
                self.fake_image_token,
                self.image_token,
                self.end_of_utterance_token,
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
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
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=False, **kwargs
        )
        # Read image_seq_len from processor_config.json if available
        init_kwargs = {}
        proc_cfg = Path(pretrained_model_name_or_path) / "processor_config.json"
        if proc_cfg.exists():
            with open(proc_cfg) as f:
                cfg = json.load(f)
            if "image_seq_len" in cfg:
                init_kwargs["image_seq_len"] = cfg["image_seq_len"]
        return cls(image_processor=image_processor, tokenizer=tokenizer, **init_kwargs)

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
        image_seq_len: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        image_seq_len = (
            image_seq_len if image_seq_len is not None else self.image_seq_len
        )
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        return_tensors = kwargs.pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
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

            images = [
                [load_image(im) if is_url(im) else im for im in sample]
                for sample in images
            ]

            # Separate image kwargs — always request row/col info for tiling
            images_kwargs = {"return_row_col_info": True}
            for k in list(kwargs.keys()):
                if k in ("return_row_col_info",):
                    images_kwargs[k] = kwargs.pop(k)

            image_inputs = self.image_processor(images, **images_kwargs)
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                image_rows = inputs.pop(
                    "rows",
                    [[0] * n_images for n_images in n_images_in_text],
                )
                image_cols = inputs.pop(
                    "cols",
                    [[0] * n_images for n_images in n_images_in_text],
                )

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []
                for sample, sample_rows, sample_cols in zip(
                    text, image_rows, image_cols
                ):
                    image_prompt_strings = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        image_prompt_strings.append(image_prompt_string)

                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError(
                            "The image token should be present in the text."
                        )

                    sample = split_sample[0]
                    for i, image_prompt_string in enumerate(image_prompt_strings):
                        sample += image_prompt_string + split_sample[i + 1]
                    prompt_strings.append(sample)

                text_inputs = self.tokenizer(prompt_strings, **kwargs)
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **kwargs)
            inputs.update(text_inputs)

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


__all__ = ["Idefics3Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("idefics3", Idefics3Processor)
