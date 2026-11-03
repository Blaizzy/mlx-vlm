"""NumPy/PIL video preprocessing matching Muse Spark's temporal patch layout."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from PIL import Image
from transformers.image_processing_utils import ImageProcessingMixin

from .image_processing import _find_resize_target


@dataclass
class MuseSparkVideoProcessor(ImageProcessingMixin):
    model_input_names: ClassVar = ["pixel_values_videos", "video_grid_thw"]
    sample_frames_in_loader: ClassVar = True

    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    max_video_frame_tokens: int = 256
    num_frames: int = 600
    fps: float = 2.0
    frame_timestamp_format: str = "Time: {timestamp:.2f}s"
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        if (
            min(
                self.patch_size,
                self.temporal_patch_size,
                self.merge_size,
                self.max_video_frame_tokens,
                self.num_frames,
            )
            <= 0
        ):
            raise ValueError(
                "Video patch sizes and token/frame limits must be positive."
            )
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video fps must be positive and finite.")

    def video_sampling_defaults(self):
        # Allow a single frame; temporal padding happens after decoding.
        return {
            "fps": self.fps,
            "min_frames": 1,
            "max_frames": self.num_frames,
            "frame_factor": 1,
        }

    def sample_frames(self, metadata, fps=None, max_frames=None):
        """Uniform sampling, with the floor rounding used by upstream."""
        source_fps = metadata.fps if metadata.fps is not None else 24.0
        rate = self.fps if fps is None else fps
        limit = self.num_frames if max_frames is None else max_frames
        total = metadata.total_num_frames
        if (
            total <= 0
            or limit <= 0
            or not np.isfinite(rate)
            or rate <= 0
            or not np.isfinite(source_fps)
            or source_fps <= 0
        ):
            raise ValueError(
                "Video sampling requires positive frame counts and finite positive fps."
            )
        count = min(int(total * rate / source_fps), limit, total)
        count = max(
            min(self.temporal_patch_size, limit, total),
            count // self.temporal_patch_size * self.temporal_patch_size,
        )
        return np.linspace(0, total - 1, int(count)).astype(np.int64)

    @staticmethod
    def _clips(videos):
        if isinstance(videos, np.ndarray):
            return list(videos) if videos.ndim == 5 else [videos]
        videos = list(videos) if not isinstance(videos, (str, bytes)) else [videos]
        if videos and (
            isinstance(videos[0], Image.Image) or getattr(videos[0], "ndim", None) == 3
        ):
            return [videos]
        return videos

    @staticmethod
    def _frames(video):
        if isinstance(video, (str, bytes)):
            raise TypeError("Use prepare_inputs() to decode video paths.")
        frames = []
        for frame in video:
            if isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
                # The shared decoder supplies (T, C, H, W); also accept THWC.
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Video frames must be uint8 RGB arrays or PIL images.")
            frames.append(frame)
        if not frames or any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("A video must contain nonempty frames of one size.")
        return frames

    @staticmethod
    def _timestamps(count, metadata, fps):
        if isinstance(metadata, dict):
            explicit = metadata.get("timestamps")
            indices = metadata.get("frames_indices", list(range(count)))
            source_fps = metadata.get("fps")
        else:
            indices = getattr(metadata, "frames_indices", None)
            # Compute timestamps from source indices ourselves. Some metadata
            # classes expose a timestamps property that fails when fps is None.
            explicit = (
                getattr(metadata, "timestamps", None)
                if metadata is not None and indices is None
                else None
            )
            if indices is None:
                indices = list(range(count))
            source_fps = getattr(metadata, "fps", None)
        if explicit is not None:
            timestamps = np.asarray(explicit, dtype=np.float64)
        else:
            rate = source_fps if source_fps is not None else fps
            if rate is None or not np.isfinite(rate) or rate <= 0:
                raise ValueError("Video fps must be positive and finite.")
            timestamps = np.asarray(indices, dtype=np.float64) / rate
        if timestamps.shape != (count,):
            raise ValueError("Video frame indices must match the decoded frame count.")
        if (
            not np.all(np.isfinite(timestamps))
            or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError(
                "Video timestamps must be finite, nonnegative, and ordered."
            )
        return timestamps

    def _preprocess(self, frames):
        height, width = frames[0].shape[:2]
        side = self.patch_size * self.merge_size
        (height, width), _ = _find_resize_target(
            height,
            width,
            patch_h=side,
            patch_w=side,
            min_tokens=1,
            max_tokens=self.max_video_frame_tokens,
        )
        pixels = (
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(frame).resize(
                            (width, height), Image.Resampling.LANCZOS
                        ),
                        dtype=np.float32,
                    )
                    for frame in frames
                ]
            )
            / 255.0
        )
        pixels = (pixels - np.asarray(self.image_mean, dtype=np.float32)) / np.asarray(
            self.image_std, dtype=np.float32
        )
        pixels = pixels.transpose(0, 3, 1, 2)
        temporal, patch = self.temporal_patch_size, self.patch_size
        if padding := -len(pixels) % temporal:
            pixels = np.concatenate([pixels, np.repeat(pixels[-1:], padding, axis=0)])
        grid_t, grid_h, grid_w = (
            len(pixels) // temporal,
            height // patch,
            width // patch,
        )
        pixels = pixels.reshape(grid_t, temporal, 3, grid_h, patch, grid_w, patch)
        # Per patch: temporal, channel, patch row, patch column.
        pixels = pixels.transpose(0, 3, 5, 1, 2, 4, 6)
        return pixels.reshape(grid_t * grid_h * grid_w, temporal * 3 * patch**2), [
            grid_t,
            grid_h,
            grid_w,
        ]

    def __call__(self, videos, video_metadata=None, fps=None):
        clips = self._clips(videos)
        if not clips:
            raise ValueError("Video input contains no clips.")
        metadata = [None] * len(clips) if video_metadata is None else video_metadata
        rates = (
            fps
            if isinstance(fps, (list, tuple))
            else [fps if fps is not None else self.fps] * len(clips)
        )
        if len(metadata) != len(clips) or len(rates) != len(clips):
            raise ValueError("Expected one metadata entry and fps value per video.")
        pixels, grids, timestamps = [], [], []
        for clip, meta, rate in zip(clips, metadata, rates):
            frames = self._frames(clip)
            times = self._timestamps(len(frames), meta, rate)
            patches, grid = self._preprocess(frames)
            pixels.append(patches)
            grids.append(grid)
            timestamps.append(times[:: self.temporal_patch_size].tolist())
        return {
            "pixel_values_videos": np.concatenate(pixels),
            "video_grid_thw": np.asarray(grids, dtype=np.int64),
            "timestamps": timestamps,
        }

    def to_dict(self):
        data = super().to_dict()
        data.pop("image_processor_type", None)
        data["video_processor_type"] = type(self).__name__
        return data
