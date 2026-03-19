"""
Processor class for Gemma3.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/processing_gemma3.py
"""

import re
from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


class Gemma3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = (
            f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"
        )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
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
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)

            # Separate return_row_col_info from kwargs if present
            images_kwargs = {}
            for k in list(kwargs.keys()):
                if k in (
                    "do_convert_rgb",
                    "do_pan_and_scan",
                    "pan_and_scan_min_crop_size",
                    "pan_and_scan_max_num_crops",
                    "pan_and_scan_min_ratio_to_activate",
                ):
                    images_kwargs[k] = kwargs.pop(k)

            image_inputs = self.image_processor(images, **images_kwargs)

            # Create empty text to be replaced with placeholders
            if not text:
                text = [
                    " ".join([self.boi_token] * len(imgs)) for imgs in batched_images
                ]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            num_crops = image_inputs.pop("num_crops", None)
            if num_crops is not None:
                if hasattr(num_crops, "tolist"):
                    num_crops = num_crops.tolist()
                if isinstance(num_crops, list) and len(num_crops) > 0:
                    # Flatten num_crops into per-batch lists
                    crop_iter = iter(num_crops)
                    batch_num_crops = [
                        [next(crop_iter) for _ in range(len(imgs))]
                        for imgs in batched_images
                    ]
                else:
                    batch_num_crops = [[0] * len(imgs) for imgs in batched_images]
            else:
                batch_num_crops = [[0] * len(imgs) for imgs in batched_images]

            for batch_idx, (prompt, imgs, crops) in enumerate(
                zip(text, batched_images, batch_num_crops)
            ):
                image_indexes = [
                    m.start() for m in re.finditer(re.escape(self.boi_token), prompt)
                ]

                if len(imgs) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(imgs)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = (
                            prompt[:idx]
                            + formatted_image_text
                            + prompt[idx + len(self.boi_token) :]
                        )
                        text[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [
                prompt.replace(self.boi_token, self.full_image_sequence)
                for prompt in text
            ]

        return_tensors = kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **kwargs)

        # Add token type ids manually
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        image_processor_input_names = [
            name for name in image_processor_input_names if name != "num_crops"
        ]
        return list(tokenizer_input_names + image_processor_input_names)

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
            for k in ("image_seq_length",):
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


__all__ = ["Gemma3Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("gemma3", Gemma3Processor)
