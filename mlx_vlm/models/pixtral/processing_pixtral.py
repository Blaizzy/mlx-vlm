"""
Processor class for Pixtral.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/processing_pixtral.py
"""

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class PixtralProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "spatial_merge_size"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        image_token: str = "[IMG]",
        image_break_token: str = "[IMG_BREAK]",
        image_end_token: str = "[IMG_END]",
        chat_template=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(
            self.image_break_token
        )
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
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
            raise ValueError("You must provide either text or images.")

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                images = [images]

            # Load images if URLs
            images = [
                [load_image(im) if is_url(im) else im for im in sample]
                for sample in images
            ]

            image_inputs = self.image_processor(images)

            if text is not None:
                image_sizes = image_inputs.get("image_sizes", [])
                # Replace [IMG] tokens with expanded sequences
                prompt_strings = []
                for batch_idx, sample in enumerate(text):
                    if self.image_token in sample:
                        sample_images = (
                            images[batch_idx] if batch_idx < len(images) else []
                        )
                        sample_sizes = (
                            image_sizes[batch_idx]
                            if batch_idx < len(image_sizes)
                            else []
                        )
                        # Normalize: slow processor returns [(h,w)] flat,
                        # ensure it's a list of tuples
                        if sample_sizes and not isinstance(
                            sample_sizes[0], (list, tuple)
                        ):
                            sample_sizes = [sample_sizes]
                        parts = sample.split(self.image_token)
                        new_sample = parts[0]
                        for img_idx in range(len(parts) - 1):
                            if img_idx < len(sample_sizes):
                                h, w = sample_sizes[img_idx]
                                num_h = h // (self.patch_size * self.spatial_merge_size)
                                num_w = w // (self.patch_size * self.spatial_merge_size)
                                # Build image token grid
                                img_tokens = ""
                                for row in range(num_h):
                                    img_tokens += self.image_token * num_w
                                    if row < num_h - 1:
                                        img_tokens += self.image_break_token
                                img_tokens += self.image_end_token
                                new_sample += img_tokens
                            else:
                                new_sample += self.image_token
                            new_sample += parts[img_idx + 1]
                        prompt_strings.append(new_sample)
                    else:
                        prompt_strings.append(sample)
                text = prompt_strings

        return_tensors = kwargs.pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            data = {**text_inputs, **image_inputs}
        else:
            data = image_inputs

        return BatchFeature(data=to_mlx(data))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + ["image_sizes"]
            )
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

        # Read processor_config.json for correct patch_size, spatial_merge_size
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        ip_overrides = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in (
                "patch_size",
                "spatial_merge_size",
                "image_token",
                "image_break_token",
                "image_end_token",
            ):
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


__all__ = ["PixtralProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("pixtral", PixtralProcessor)
