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

import copy
from functools import partial

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, infer_channel_dimension_format
from transformers.processing_utils import VideosKwargs
from transformers.utils import logging
from transformers.video_utils import (
    VideoMetadata,
    convert_to_rgb,
    is_valid_video,
    make_batched_metadata,
    make_batched_videos,
)

from ..gemma4.processing_gemma4 import Gemma4VideoProcessor
from .image_processing_embedding_gemma2 import EmbeddingGemma2ImageProcessor

logger = logging.get_logger(__name__)


class EmbeddingGemma2VideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int
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

    def __init__(
        self,
        fps=1,
        max_frames=32,
        overflow_strategy="uniform",
        add_timestamps=False,
        do_sample_frames=True,
        num_frames=None,
        **kwargs,
    ):
        kwargs.setdefault("do_normalize", True)
        super().__init__(num_frames=num_frames, **kwargs)
        self.fps = fps
        self.max_frames = max_frames
        self.overflow_strategy = overflow_strategy
        self.add_timestamps = add_timestamps
        self.do_sample_frames = do_sample_frames
        self.do_convert_rgb = kwargs.get("do_convert_rgb", True)
        self.resample = kwargs.get("resample", 3)
        self.return_metadata = kwargs.get("return_metadata", False)

    def video_sampling_defaults(self):
        return {"fps": self.fps, "max_frames": self.max_frames}

    def __call__(self, videos, **kwargs):
        return self.preprocess(videos, **kwargs)

    def preprocess(
        self,
        videos,
        video_metadata=None,
        return_metadata=None,
        return_tensors=None,
        **kwargs,
    ):
        videos = make_batched_videos(videos)
        metadata = copy.deepcopy(make_batched_metadata(videos, video_metadata))
        options = {
            name: getattr(self, name)
            for name in (
                "fps",
                "max_frames",
                "overflow_strategy",
                "num_frames",
            )
        }
        options.update({name: kwargs[name] for name in options if name in kwargs})
        do_sample = kwargs.get("do_sample_frames", self.do_sample_frames)
        sampler = partial(self.sample_frames, **options) if do_sample else None
        image_processor = EmbeddingGemma2ImageProcessor(**self.__dict__)
        pixels, positions, counts, frame_counts = [], [], [], []
        for index, video in enumerate(videos):
            if is_valid_video(video):
                video = np.asarray(video)
                if sampler is not None:
                    indices = sampler(metadata=metadata[index])
                    metadata[index].frames_indices = indices
                    video = video[indices]
            elif isinstance(video, list):
                if do_sample:
                    raise ValueError(
                        "Sampling frames from a list of images is not supported! Set `do_sample_frames=False`."
                    )
                video = image_processor.fetch_images(video)
            else:
                video, metadata[index] = self._decode_video(video, sampler)
            video = np.asarray(video)
            layout = kwargs.get("input_data_format") or infer_channel_dimension_format(
                video, num_channels=(1, 3, 4)
            )
            if kwargs.get("do_convert_rgb", self.do_convert_rgb):
                video = convert_to_rgb(video, input_data_format=layout)
                layout = ChannelDimension.FIRST
            image_kwargs = kwargs | {"input_data_format": layout}
            inputs = image_processor(list(video), return_tensors="np", **image_kwargs)
            pixels.append(inputs["pixel_values"])
            positions.append(inputs["image_position_ids"])
            counts.append(int(inputs["num_soft_tokens_per_image"][0]))
            frame_counts.append(len(video))
        result = BatchFeature(
            {
                "pixel_values_videos": np.concatenate(pixels),
                "video_position_ids": np.concatenate(positions),
                "num_frames_per_video": frame_counts,
                "num_soft_tokens_per_video": counts,
            },
            tensor_type=return_tensors,
        )
        include_metadata = (
            self.return_metadata if return_metadata is None else return_metadata
        )
        if include_metadata:
            result["video_metadata"] = metadata
        return result

    @staticmethod
    def _decode_video(path, sampler):
        from ._video_decoder import VideoDecodeUnavailable, decode_video

        try:
            return decode_video(path, sampler)
        except VideoDecodeUnavailable as exc:
            logger.warning_once(
                f"Using OpenCV video decoding: {exc}. Colors and frame timing may "
                "differ from the reference decoder."
            )

        from mlx_vlm.utils import load_video

        def sample(metadata, **kwargs):
            return (
                sampler(metadata=metadata)
                if sampler is not None
                else np.arange(metadata.total_num_frames)
            )

        video, info = load_video(str(path), frame_sampler=sample)
        metadata = VideoMetadata(
            total_num_frames=info.total_num_frames,
            fps=info.fps,
            duration=info.duration,
            frames_indices=info.frames_indices,
            width=info.width,
            height=info.height,
            video_backend="opencv",
        )
        return video.transpose(0, 2, 3, 1), metadata

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
