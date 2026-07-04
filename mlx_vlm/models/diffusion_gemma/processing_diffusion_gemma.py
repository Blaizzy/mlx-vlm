"""
Processor for DiffusionGemma4 checkpoints.

Reuses the MLX-native Gemma4 processor (image processor + tokenizer + image
token expansion + ``mm_token_type_ids``), so no torch/torchvision is pulled in.
Audio inputs are rejected: this model port supports text, images, and videos.
"""

from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import materialize_mx_arrays
from ..gemma4.processing_gemma4 import Gemma4Processor

_TOOL_PARSER_TOKENS = {
    "<|tool_call>",
    "<tool_call|>",
    '<|"|>',
    "<|channel>",
    "<channel|>",
}

_TOOL_PARSER_TOKEN_ATTRIBUTES = (
    "stc_token",
    "etc_token",
    "escape_token",
    "soc_token",
    "eoc_token",
)


def _token_text(token):
    return getattr(token, "content", token)


def _make_tool_parser_tokens_non_special(tokenizer):
    for attr in _TOOL_PARSER_TOKEN_ATTRIBUTES:
        if _token_text(getattr(tokenizer, attr, None)) in _TOOL_PARSER_TOKENS:
            try:
                setattr(tokenizer, attr, None)
            except AttributeError:
                pass

    additional_tokens = getattr(tokenizer, "additional_special_tokens", None)
    if not additional_tokens:
        return
    tokenizer.additional_special_tokens = [
        token
        for token in additional_tokens
        if _token_text(token) not in _TOOL_PARSER_TOKENS
    ]


class DiffusionGemma4Processor(Gemma4Processor):
    model_type = "diffusion_gemma"
    image_processor_class = "Gemma4ImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "Gemma4VideoProcessor"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _make_tool_parser_tokens_non_special(self.tokenizer)

    @staticmethod
    def _flatten_visual_items(values):
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]

        items = []
        for value in values:
            if getattr(value, "ndim", 0) == 4:
                items.extend(value[i] for i in range(value.shape[0]))
            else:
                items.append(value)
        return items

    @staticmethod
    def _stack_visual_items(items):
        if not items:
            return None
        shapes = {tuple(item.shape) for item in items}
        if len(shapes) != 1:
            return items
        if any(isinstance(item, mx.array) for item in items):
            return mx.stack(
                [
                    item if isinstance(item, mx.array) else mx.array(item)
                    for item in items
                ]
            )
        return np.stack(items)

    @staticmethod
    def _visual_block_types(mm_token_type_ids):
        if mm_token_type_ids is None:
            return []
        if isinstance(mm_token_type_ids, mx.array):
            rows = mm_token_type_ids.tolist()
        elif isinstance(mm_token_type_ids, np.ndarray):
            rows = mm_token_type_ids.tolist()
        else:
            rows = mm_token_type_ids

        block_types = []
        for row in rows:
            previous = 0
            for value in row:
                value = int(value)
                if value in (1, 2) and value != previous:
                    block_types.append(value)
                previous = value
        return block_types

    def _merge_video_pixel_values(self, inputs: BatchFeature) -> BatchFeature:
        pixel_values_videos = inputs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            return inputs

        image_items = self._flatten_visual_items(inputs.pop("pixel_values", None))
        video_items = self._flatten_visual_items(pixel_values_videos)
        if not image_items:
            inputs["pixel_values"] = self._stack_visual_items(video_items)
            return inputs

        merged_items = []
        block_types = self._visual_block_types(inputs.get("mm_token_type_ids"))
        for block_type in block_types:
            if block_type == 1 and image_items:
                merged_items.append(image_items.pop(0))
            elif block_type == 2 and video_items:
                merged_items.append(video_items.pop(0))
        merged_items.extend(image_items)
        merged_items.extend(video_items)
        inputs["pixel_values"] = self._stack_visual_items(merged_items)
        return inputs

    def __call__(
        self,
        images=None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if audio is not None:
            raise ValueError("DiffusionGemma4 audio inputs are not supported yet.")
        if text is None and images is None and videos is None:
            raise ValueError(
                "Provide `text`, `images`, and/or `videos` for DiffusionGemma4Processor."
            )
        inputs = super().__call__(images=images, text=text, videos=videos, **kwargs)
        return materialize_mx_arrays(self._merge_video_pixel_values(inputs))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import warnings

        # The gemma4 processor builds an audio feature extractor whose mel
        # filter construction warns for this checkpoint's settings; audio is
        # rejected by this processor, so the warning is pure noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mel filter.*")
            processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        _make_tool_parser_tokens_non_special(processor.tokenizer)
        return processor


__all__ = ["DiffusionGemma4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("diffusion_gemma", DiffusionGemma4Processor)
