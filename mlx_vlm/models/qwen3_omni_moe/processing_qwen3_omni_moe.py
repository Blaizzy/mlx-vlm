"""
Processor class for Qwen3OmniMoe.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
"""

import re
from typing import Optional

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import TextInput

from ..base import to_mlx


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the audio encoder.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class Qwen3OmniMoeProcessor(ProcessorMixin):
    attributes = [
        "image_processor",
        "video_processor",
        "feature_extractor",
        "tokenizer",
    ]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor,
            video_processor,
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
        )
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: Optional[TextInput] = None,
        images: Optional[ImageInput] = None,
        videos=None,
        audio=None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        seconds_per_chunk = kwargs.pop("seconds_per_chunk", 2.0)
        position_id_per_seconds = kwargs.pop("position_id_per_seconds", 13.0)
        use_audio_in_video = kwargs.pop("use_audio_in_video", False)
        fps = kwargs.pop("fps", 1.0)

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
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask", None
            )
            audio_inputs["input_features"] = audio_inputs.pop("input_features", None)
            audio_lengths = iter(
                _get_feat_extract_output_lengths(
                    audio_inputs["feature_attention_mask"].sum(-1)
                )
            )
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images)
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos)
            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps
            ] * len(videos_inputs.get("video_grid_thw", []))
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        return_tensors = kwargs.pop("return_tensors", None)
        texts_inputs = self.tokenizer(text, **kwargs)

        return BatchFeature(
            data=to_mlx(
                {
                    **texts_inputs,
                    **images_inputs,
                    **videos_inputs,
                    **audio_inputs,
                }
            ),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            special_tokens = [
                re.escape(tok)
                for tok in [
                    self.audio_token,
                    self.image_token,
                    self.video_token,
                ]
            ]
            pattern = "|".join(special_tokens)
            positions = sorted(
                [
                    (match.start(), match.group())
                    for match in re.finditer(pattern, sample)
                ]
            )
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * next(audio_lengths),
                        1,
                    )
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(
                        self.image_token,
                        "<|image_placeholder|>" * image_seq_length,
                        1,
                    )
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = (
                            next(video_grid_thw).prod() // merge_length_video
                        )
                        sample = sample.replace(
                            self.video_token,
                            "<|video_placeholder|>" * video_seq_length,
                            1,
                        )
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = (
                            curr_video_grid_thw[1] // self.video_processor.merge_size
                        )
                        width = (
                            curr_video_grid_thw[2] // self.video_processor.merge_size
                        )
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(
                            -1, 1, 1
                        )
                        video_token_indices = np.broadcast_to(
                            video_token_indices,
                            (video_token_indices.shape[0], height, width),
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices
                            * next(video_second_per_grid)
                            * position_id_per_seconds
                        )

                        video_data_index, audio_data_index = 0, 0
                        placeholder_string = (
                            self.vision_bos_token + self.audio_bos_token
                        )
                        while video_data_index < len(
                            video_token_indices
                        ) and audio_data_index < len(audio_token_indices):
                            if (
                                video_token_indices[video_data_index]
                                <= audio_token_indices[audio_data_index]
                            ):
                                placeholder_string += "<|video_placeholder|>"
                                video_data_index += 1
                            else:
                                placeholder_string += "<|audio_placeholder|>"
                                audio_data_index += 1
                        if video_data_index < len(video_token_indices):
                            placeholder_string += "<|video_placeholder|>" * (
                                len(video_token_indices) - video_data_index
                            )
                        if audio_data_index < len(audio_token_indices):
                            placeholder_string += "<|audio_placeholder|>" * (
                                len(audio_token_indices) - audio_data_index
                            )
                        placeholder_string += (
                            self.audio_eos_token + self.vision_eos_token
                        )
                        sample = sample.replace(
                            self.vision_bos_token
                            + self.video_token
                            + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen3OmniMoeProcessor"]
