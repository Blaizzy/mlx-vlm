from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .constants import MINIMAX_H3_TEXT_ENCODER_LAYER
from .processing import (
    process_qwen_images,
    process_qwen_videos,
    sample_reference_video_frames,
)
from .prompting import build_fl2va_presentation, build_ref2va_presentation
from .references import MiniMaxH3PreparedReference


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConditioningOutput:
    hidden_states: mx.array
    token_tags: mx.array
    input_ids: mx.array
    image_grid_thw: mx.array | None = None
    video_grid_thw: mx.array | None = None


class MiniMaxH3Conditioner:
    """H3 adapter for the existing MLX Qwen3-VL vision/language model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
        has_vision: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.has_vision = has_vision

    def _encode(
        self,
        token_ids: list[int],
        token_tags: list[int],
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        pixel_values_videos: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
    ) -> MiniMaxH3ConditioningOutput:
        if (
            pixel_values is not None or pixel_values_videos is not None
        ) and not self.has_vision:
            raise ValueError(
                "this H3 conditioner was converted without its vision tower"
            )
        input_ids = mx.array([token_ids], dtype=mx.int32)
        hidden_states = self.model.hidden_state_at_layer(
            input_ids,
            self.layer,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return MiniMaxH3ConditioningOutput(
            hidden_states=hidden_states,
            token_tags=mx.array(token_tags, dtype=mx.int32),
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    def encode_fl2va(
        self,
        prompt: str,
        images: list[mx.array] | None = None,
    ) -> MiniMaxH3ConditioningOutput:
        pixel_values = image_grid = None
        image_token_counts: list[int] = []
        if images:
            pixel_values, image_grid = process_qwen_images(images)
            image_token_counts = [
                math_product(grid) // 4 for grid in image_grid.tolist()
            ]
        token_ids, token_tags = build_fl2va_presentation(
            self.tokenizer, prompt, image_token_counts
        )
        return self._encode(
            token_ids,
            token_tags,
            pixel_values=pixel_values,
            image_grid_thw=image_grid,
        )

    def encode_ref2va(
        self,
        prompt: str,
        references: list[MiniMaxH3PreparedReference],
    ) -> MiniMaxH3ConditioningOutput:
        images = [
            reference.image for reference in references if reference.kind == "image"
        ]
        pixel_values = image_grid = None
        image_token_counts: list[int] = []
        if images:
            pixel_values, image_grid = process_qwen_images(images)
            image_token_counts = [
                math_product(grid) // 4 for grid in image_grid.tolist()
            ]

        videos = [reference for reference in references if reference.kind == "video"]
        pixel_values_videos = video_grid = None
        video_block_token_counts: list[int] = []
        if videos:
            sampled_videos = []
            for reference in videos:
                frames, timestamps = sample_reference_video_frames(reference.frames)
                reference.block_timestamps = timestamps
                sampled_videos.append(frames)
            pixel_values_videos, video_grid = process_qwen_videos(sampled_videos)
            video_block_token_counts = [
                grid[1] * grid[2] // 4 for grid in video_grid.tolist()
            ]
            for reference, grid in zip(videos, video_grid.tolist()):
                if grid[0] != len(reference.block_timestamps):
                    raise ValueError(
                        "Qwen video blocks and MiniMax-H3 timestamp labels disagree"
                    )

        token_ids, token_tags = build_ref2va_presentation(
            self.tokenizer,
            prompt,
            references,
            image_token_counts,
            video_block_token_counts,
        )
        return self._encode(
            token_ids,
            token_tags,
            pixel_values=pixel_values,
            image_grid_thw=image_grid,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid,
        )


def math_product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = ["MiniMaxH3Conditioner", "MiniMaxH3ConditioningOutput"]
