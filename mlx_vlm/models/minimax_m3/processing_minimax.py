# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
MiniMax VL family HuggingFace-compatible Processor, ImageProcessor, VideoProcessor.
"""

import math
import re
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos


class MiniMaxVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "videos_kwargs": {
            "do_resize": False,
            "return_metadata": True,
        },
    }


class MiniMaxVLProcessor(ProcessorMixin):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def __init__(
        self, image_processor=None, tokenizer=None, video_processor=None, **kwargs
    ):
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN)
        super().__init__(image_processor, tokenizer, video_processor)
        # Video expansion also uses image start/end tokens. Separate video
        # start/end tokens exist in the tokenizer, but the original MiniMax
        # serving path did not use them; keep that behavior for compatibility.
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_START_TOKEN
        )
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_END_TOKEN
        )

    def _prune_video_tokens(
        self,
        input_text: str,
        video_segments: List[int],
        video_token: str,
    ) -> str:
        """
        Prune video tokens by temporal_patch_size (e.g., 2:1).

        Expects the prompt to carry exactly sum(video_segments) video
        tokens — i.e. one token per *sampled* frame. Then drops token.

        Args:
            input_text: prompt with N video_tokens per segment
            video_segments: actual sampled frame count per video segment
            video_token: the video token string, e.g. ']<]video[>['

        Returns:
            Pruned input_text with ~N/temporal_patch_size tokens per segment.
        """
        # If no videos or temporal_patch_size <= 1, no pruning needed
        if not video_segments or self.video_processor.temporal_patch_size <= 1:
            return input_text

        # Split while keeping delimiters
        special_tokens = [video_token]  # , image_token]
        pattern = "|".join(map(re.escape, special_tokens))
        parts = re.split(f"({pattern})", input_text)

        def is_timestamp(text: str) -> bool:
            """Check if text ends with timestamp format like ']<]0.0 seconds[>['"""
            return (
                text.endswith("seconds[>[")
                or text.endswith("seconds[>[ ")
                or text.endswith("seconds [>[")
                or text.endswith("seconds [>[ ")
            )

        def extract_timestamp(text: str) -> str:
            """Extract timestamp text from the end, starting from ']<]'"""
            start_index = text.rfind("]<]")
            if start_index == -1:
                raise ValueError(f"Failed to extract timestamp: {text}")
            return text[start_index:]

        # Build new text with pruned video tokens
        final_parts = []
        current_seg_idx = 0  # Which video segment we're in
        frame_in_seg = 0  # Frame index within current segment
        last_timestamp_len = 0  # Length of timestamp to potentially remove

        for part in parts:
            if part == video_token:
                if current_seg_idx < len(video_segments):
                    if frame_in_seg % self.video_processor.temporal_patch_size == 0:
                        # Keep this video token
                        final_parts.append(part)
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        last_timestamp_len = 0
                    else:
                        # Skip this video token
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        # Remove the timestamp that was already appended
                        if last_timestamp_len > 0:
                            # Truncate the last part to remove timestamp
                            assert len(final_parts) > 0
                            final_parts[-1] = final_parts[-1][:-last_timestamp_len]
                            last_timestamp_len = 0
                else:
                    # No more video segments, keep as is
                    final_parts.append(part)
                    last_timestamp_len = 0
            else:
                # Text part
                final_parts.append(part)
                # Check if this text ends with a timestamp
                if is_timestamp(part):
                    last_timestamp_len = len(extract_timestamp(part))
                else:
                    last_timestamp_len = 0

        return "".join(final_parts)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs: Unpack[MiniMaxVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            MiniMaxVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            images_kwargs = output_kwargs["images_kwargs"]
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]

        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_kwargs = output_kwargs["videos_kwargs"]
            video_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = video_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs["video_metadata"]
        else:
            video_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        # Expand image tokens
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.IMAGE_TOKEN in text[i]:
                    num_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.IMAGE_TOKEN,
                        self.VISION_START_TOKEN
                        + placeholder * num_tokens
                        + self.VISION_END_TOKEN,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace(placeholder, self.IMAGE_TOKEN)

        # Expand video tokens
        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.VIDEO_TOKEN in text[i]:
                    metadata = video_metadata[index]
                    grid_t = video_grid_thw[index][0]
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length

                    video_placeholder = ""
                    for frame_idx in range(grid_t):
                        if (
                            metadata.fps is not None
                            and metadata.frames_indices is not None
                        ):
                            ts = (
                                metadata.frames_indices[
                                    min(
                                        frame_idx
                                        * self.video_processor.temporal_patch_size,
                                        len(metadata.frames_indices) - 1,
                                    )
                                ]
                                / metadata.fps
                            )
                            video_placeholder += f"]<]{ts:.1f} seconds[>["
                        video_placeholder += (
                            self.VISION_START_TOKEN
                            + placeholder * frame_seqlen
                            + self.VISION_END_TOKEN
                        )

                    text[i] = text[i].replace(self.VIDEO_TOKEN, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace(placeholder, self.VIDEO_TOKEN)

        # Tokenize
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )
