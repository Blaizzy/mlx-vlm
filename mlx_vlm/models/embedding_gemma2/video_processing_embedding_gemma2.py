# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""EmbeddingGemma 2 video sampling and variable-length frame batching."""

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.gemma4.video_processing_gemma4 import (
    Gemma4VideoProcessor,
    Gemma4VideoProcessorKwargs,
    convert_video_to_patches,
    pad_to_max_patches,
)
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoMetadata

logger = logging.get_logger(__name__)
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


class EmbeddingGemma2VideoProcessorKwargs(Gemma4VideoProcessorKwargs, total=False):
    add_timestamps: bool
    max_frames: int | None
    overflow_strategy: str | None


class EmbeddingGemma2VideoProcessor(Gemma4VideoProcessor):
    valid_kwargs = EmbeddingGemma2VideoProcessorKwargs
    model_input_names = [
        "pixel_values_videos",
        "video_position_ids",
        "num_frames_per_video",
    ]
    num_frames = None
    fps = 1
    max_frames = 32
    overflow_strategy = "uniform"
    add_timestamps = False

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        resample: "tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )
        max_patches = max_soft_tokens * pooling_kernel_size**2
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_video = []
        num_frames_per_video = []
        for video in videos:
            if do_convert_rgb:
                video = self.convert_to_rgb(video)
            if do_resize:
                video = self.aspect_ratio_preserving_resize(
                    video=video,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )
            video = self.rescale_and_normalize(
                video, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            num_frames = video.shape[0]
            patch_height = video.shape[-2] // patch_size
            patch_width = video.shape[-1] // patch_size
            patches = convert_video_to_patches(video, patch_size)
            num_soft_tokens_per_video.append(patches.shape[1] // pooling_kernel_size**2)
            num_frames_per_video.append(num_frames)
            device = video.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            real_positions = stacked_grid.reshape(patches.shape[1], 2)
            real_positions = real_positions[None, ...].repeat(num_frames, 1, 1)
            patches, positions = pad_to_max_patches(
                patches, real_positions, max_patches
            )
            pixel_values.append(patches)
            position_ids.append(positions)
        pixel_values = torch.cat(pixel_values, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        data = {
            "pixel_values_videos": pixel_values,
            "video_position_ids": position_ids,
            "num_frames_per_video": num_frames_per_video,
            "num_soft_tokens_per_video": num_soft_tokens_per_video,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        max_frames: int | None = None,
        overflow_strategy: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        if kwargs.get("num_frames") is not None:
            raise ValueError(
                f"Sampling with `num_frames` is not supported for {self.__class__.__name__}. Please use `fps` and `max_frames` to control video sampling."
            )
        if fps is not None and (metadata.fps is None or metadata.duration is None):
            logger.warning_once(
                "Asked to sample uniformly with `fps`, but the video metadata has no `fps` or `duration`. Keeping every frame and applying only the `max_frames` budget. Pass a `VideoMetadata` object with a valid `fps` and `duration` to sample at a target frame rate."
            )
            fps = None
        if fps is None:
            indices = np.arange(metadata.total_num_frames, dtype=int)
        else:
            step = metadata.fps / fps
            num_sampled = max(1, int(metadata.duration * fps))
            indices = np.array(
                [
                    min(metadata.total_num_frames - 1, int(i * step))
                    for i in range(num_sampled)
                ],
                dtype=int,
            )
        if overflow_strategy is not None:
            if max_frames is None:
                raise ValueError(
                    f"You must pass `max_frames` when requesting an overflow_strategy={overflow_strategy}!"
                )
            if len(indices) <= max_frames:
                pass
            elif overflow_strategy == "truncate":
                indices = indices[:max_frames]
            elif overflow_strategy == "uniform":
                linspace_idx = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
                indices = np.array([indices[i] for i in linspace_idx], dtype=int)
            else:
                raise ValueError(
                    f"You passed `overflow_strategy={overflow_strategy}` but expected one of ['truncate', 'uniform']"
                )
        return indices
