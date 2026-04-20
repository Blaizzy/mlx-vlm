"""
Processor class for Gemma3n.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/processing_gemma3n.py
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


class Gemma3nProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "audio_seq_length", "image_seq_length"]
    feature_extractor_class = "AutoFeatureExtractor"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        chat_template=None,
        audio_seq_length: int = 188,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.audio_seq_length = audio_seq_length
        self.audio_token_id = tokenizer.audio_token_id
        self.boa_token = tokenizer.boa_token
        self.audio_token = tokenizer.audio_token
        audio_tokens_expanded = "".join([tokenizer.audio_token] * audio_seq_length)
        self.full_audio_sequence = (
            f"\n\n{tokenizer.boa_token}{audio_tokens_expanded}{tokenizer.eoa_token}\n\n"
        )

        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = (
            f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"
        )

        if feature_extractor is None:
            self.feature_extractor = None
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            if chat_template is not None:
                self.chat_template = chat_template
        else:
            super().__init__(
                feature_extractor=feature_extractor,
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
        audio=None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None and audio is None:
            raise ValueError("Provide at least one of `text`, `images`, or `audio`.")

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        if audio is not None:
            audio_kwargs = {}
            for k in list(kwargs.keys()):
                if k in (
                    "sampling_rate",
                    "padding",
                    "truncation",
                    "return_attention_mask",
                ):
                    audio_kwargs[k] = kwargs.pop(k)
            audio_inputs = self.feature_extractor(audio, **audio_kwargs)

            if not text:
                text = [self.audio_token for _ in audio]

            # Expand placeholder audio tokens to the full audio token sequence
            text = [
                prompt.replace(self.audio_token, self.full_audio_sequence)
                for prompt in text
            ]
        else:
            audio_inputs = {}

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)

            images_kwargs = {}
            for k in list(kwargs.keys()):
                if k in ("do_convert_rgb",):
                    images_kwargs[k] = kwargs.pop(k)
            image_inputs = self.image_processor(batched_images, **images_kwargs)

            if not text:
                text = [
                    " ".join([self.image_token] * len(imgs)) for imgs in batched_images
                ]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Expand placeholder image tokens to the full image token sequence
            text = [
                prompt.replace(self.image_token, self.full_image_sequence)
                for prompt in text
            ]

        kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text=text, **kwargs)

        # Add token type ids manually
        array_ids = np.array(text_inputs["input_ids"])
        token_type_ids = np.zeros_like(array_ids)
        token_type_ids[array_ids == self.image_token_id] = 1
        token_type_ids[array_ids == self.audio_token_id] = 3
        text_inputs["token_type_ids"] = token_type_ids.tolist()

        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **audio_inputs})
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = [
            name for name in image_processor_input_names if name != "num_crops"
        ]
        return list(
            tokenizer_input_names
            + image_processor_input_names
            + audio_processor_input_names
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer

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
            for k in ("audio_seq_length", "image_seq_length"):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]
            ip_cfg = proc_cfg.get("image_processor", {})
            if "patch_size" in ip_cfg:
                ip_overrides["patch_size"] = ip_cfg["patch_size"]
            if "size" in ip_cfg:
                ip_overrides["size"] = ip_cfg["size"]

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
        except (ValueError, OSError, ModuleNotFoundError):
            feature_extractor = None

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
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            **proc_kwargs,
        )


__all__ = ["Gemma3nProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("gemma3n", Gemma3nProcessor)
