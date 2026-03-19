"""
Processor class for Mllama (Llama 3.2 Vision).

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/processing_mllama.py
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


def get_cross_attention_token_mask(input_ids, image_token_id):
    """
    Generate a cross-attention token mask for image tokens in the input sequence.
    """
    image_token_locations = [
        i for i, token in enumerate(input_ids) if token == image_token_id
    ]

    if len(image_token_locations) == 0:
        return []

    if len(image_token_locations) == 1:
        return [[image_token_locations[0], -1]]

    vision_masks = [
        [loc1, loc2]
        for loc1, loc2 in zip(image_token_locations[:-1], image_token_locations[1:])
    ]

    # last image will attend to all subsequent text
    vision_masks.append([image_token_locations[-1], len(input_ids)])

    # consecutive vision tokens should all attend to subsequent text
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]

    return vision_masks


def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask,
    num_tiles,
    max_num_tiles,
    length,
):
    """
    Convert the cross attention mask indices to a cross attention mask 4D array.
    """
    batch_size = len(cross_attention_token_mask)
    max_num_images = (
        max(len(masks) for masks in cross_attention_token_mask)
        if cross_attention_token_mask
        else 0
    )

    cross_attention_mask = np.zeros(
        shape=(batch_size, length, max_num_images, max_num_tiles),
        dtype=np.int64,
    )

    for sample_idx, (sample_masks, sample_num_tiles) in enumerate(
        zip(cross_attention_token_mask, num_tiles)
    ):
        for mask_idx, (locations, mask_num_tiles) in enumerate(
            zip(sample_masks, sample_num_tiles)
        ):
            if len(locations) == 2:
                start, end = locations
                end = min(end, length)
                if end == -1:
                    end = length
                cross_attention_mask[
                    sample_idx, start:end, mask_idx, :mask_num_tiles
                ] = 1

    return cross_attention_mask


def build_string_from_input(prompt, bos_token, image_token):
    """
    Builds a string from the input prompt by adding bos_token if not already present.
    """
    if bos_token in prompt:
        return prompt

    num_image_tokens_on_start = 0
    while prompt.startswith(image_token):
        prompt = prompt[len(image_token) :]
        num_image_tokens_on_start += 1

    return f"{image_token * num_image_tokens_on_start}{bos_token}{prompt}"


class MllamaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        if not hasattr(tokenizer, "image_token"):
            self.image_token = "<|image|>"
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token = tokenizer.image_token
            self.image_token_id = tokenizer.image_token_id

        self.python_token = "<|python_tag|>"
        self.python_token_id = tokenizer.convert_tokens_to_ids(self.python_token)
        self.bos_token = tokenizer.bos_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

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
            raise ValueError("You must specify either text or images.")

        return_tensors = kwargs.pop("return_tensors", None)
        max_image_tiles = kwargs.pop("max_image_tiles", 4)

        data = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (
                isinstance(text, (list, tuple))
                and all(isinstance(t, str) for t in text)
            ):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )
            n_images_in_text = [t.count(self.image_token) for t in text]
            text = [
                build_string_from_input(text_item, self.bos_token, self.image_token)
                for text_item in text
            ]
            encoding = self.tokenizer(text, **kwargs)
            n_images_in_ids = [
                token_ids.count(self.image_token_id)
                for token_ids in encoding["input_ids"]
            ]
            data.update(encoding)

        n_images_in_images = [0]
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_nested_list_of_images(images)
            n_images_in_images = [len(sample) for sample in images]

        if text is not None:
            if any(batch_img == 0 for batch_img in n_images_in_text) and not all(
                batch_img == 0 for batch_img in n_images_in_text
            ):
                raise ValueError(
                    "If a batch of text is provided, there should be either no images or at least one image per sample"
                )
            if sum(n_images_in_text) > 0 and (
                n_images_in_images != n_images_in_text
                or n_images_in_ids != n_images_in_images
            ):
                if images is None:
                    raise ValueError(
                        "No image were provided, but there are image tokens in the prompt"
                    )
                else:
                    raise ValueError(
                        f"The number of image tokens in each text ({n_images_in_text}) should be the same as the "
                        f"number of provided images per batch ({n_images_in_images})."
                    )

        if images is not None:
            image_features = self.image_processor(
                images, max_image_tiles=max_image_tiles
            )
            num_tiles = image_features.pop("num_tiles")
            data.update(image_features)

        # Create cross attention mask
        if images is not None and text is not None:
            cross_attention_token_mask = [
                get_cross_attention_token_mask(token_ids, self.image_token_id)
                for token_ids in encoding["input_ids"]
            ]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=self.image_processor.max_image_tiles,
                length=max(len(input_ids) for input_ids in encoding["input_ids"]),
            )
            data["cross_attention_mask"] = cross_attention_mask

        return BatchFeature(data=to_mlx(data))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        image_processor_input_names = [
            name for name in image_processor_input_names if name != "num_tiles"
        ]
        return list(
            tokenizer_input_names
            + image_processor_input_names
            + ["cross_attention_mask"]
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

        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
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
        )


__all__ = ["MllamaProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("mllama", MllamaProcessor)
