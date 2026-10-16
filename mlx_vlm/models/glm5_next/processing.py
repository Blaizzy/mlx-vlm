import json
import math
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.image_transforms import resize as resize_image
from transformers.image_utils import ChannelDimension, PILImageResampling
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template
from ..qwen3_vl.processing_qwen3_vl import _flatten_images


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
):
    """Return the aligned GLM-5-Next canvas under a token budget."""
    if min(num_frames, height, width, temporal_factor, factor) <= 0:
        raise ValueError("Image dimensions and alignment factors must be positive.")
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError("Expected 0 < min_pixels <= max_pixels.")

    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, factor):
        return math.ceil(value / factor) * factor

    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )

    def fit_within_budget():
        minimum = aligned_frames * factor**2
        if max_pixels < minimum:
            raise ValueError(
                f"max_pixels={max_pixels} is too small; at least {minimum} is required."
            )
        low, high = 1, height
        best = (factor, factor)
        while low <= high:
            content_h = (low + high) // 2
            content_w = max(1, math.floor(width * content_h / height))
            candidate = (align(content_h, factor), align(content_w, factor))
            if aligned_frames * candidate[0] * candidate[1] <= max_pixels:
                best = candidate
                low = content_h + 1
            else:
                high = content_h - 1
        return best

    aligned_h = align(height, factor)
    aligned_w = align(width, factor)
    pixels = aligned_frames * aligned_h * aligned_w
    if pixels < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_h = align(max(1, math.ceil(height * scale)), factor)
        aligned_w = align(max(1, math.ceil(width * scale)), factor)
        pixels = aligned_frames * aligned_h * aligned_w
    if pixels > max_pixels:
        aligned_h, aligned_w = fit_within_budget()
    return aligned_h, aligned_w


def _resize_geometry(
    num_frames,
    height,
    width,
    *,
    patch_size,
    merge_size,
    temporal_patch_size,
    patch_expand_factor,
    min_image_tokens,
    max_image_tokens,
):
    factor = patch_size * merge_size * patch_expand_factor
    pixels_per_token = temporal_patch_size * factor**2
    target_h, target_w = smart_resize(
        num_frames,
        height,
        width,
        temporal_factor=temporal_patch_size,
        factor=factor,
        min_pixels=min_image_tokens,
        max_pixels=max_image_tokens,
    )
    scale = min(target_h / height, target_w / width)
    if num_frames * height * width >= min_image_tokens * pixels_per_token:
        scale = min(1.0, scale)
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    return target_h, target_w, content_h, content_w


def _to_numpy_processor_image(image, convert_rgb=True):
    """Convert an image to channel-first NumPy without requiring Torch."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if hasattr(image, "convert"):
        if convert_rgb:
            image = image.convert("RGB")
        array = np.asarray(image)
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = array.transpose(2, 0, 1)
    if convert_rgb and array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif convert_rgb and array.shape[0] == 4:
        array = array[:3]
    return array


def _resize_frames_pil(frames, target_h, target_w, resample):
    """Match the official torch-free Transformers PIL resize backend."""
    if frames.shape[-2:] == (target_h, target_w):
        return frames
    return np.stack(
        [
            resize_image(
                frame,
                (target_h, target_w),
                resample=resample,
                data_format=ChannelDimension.FIRST,
                input_data_format=ChannelDimension.FIRST,
            )
            for frame in frames
        ]
    )


class Glm5NextImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        resample=PILImageResampling.BICUBIC,
        size=None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]
        self.do_resize = do_resize
        self.resample = resample
        self.size = size or {"longest_edge": 1}
        self.default_to_square = False
        self.do_convert_rgb = do_convert_rgb

    def _settings(self, kwargs):
        return {
            name: kwargs.get(name, getattr(self, name))
            for name in (
                "patch_size",
                "temporal_patch_size",
                "merge_size",
                "patch_expand_factor",
                "min_image_tokens",
                "max_image_tokens",
            )
        }

    def _process_one(self, image, **kwargs):
        image = _to_numpy_processor_image(
            image, kwargs.get("do_convert_rgb", self.do_convert_rgb)
        )
        channels, height, width = image.shape
        settings = self._settings(kwargs)
        if kwargs.get("do_resize", self.do_resize):
            target_h, target_w, content_h, content_w = _resize_geometry(
                settings["temporal_patch_size"],
                height,
                width,
                **settings,
            )
            image = _resize_frames_pil(
                image[None],
                content_h,
                content_w,
                kwargs.get("resample", self.resample),
            )[0]
            image = np.pad(
                image,
                ((0, 0), (0, target_h - content_h), (0, target_w - content_w)),
            )
        else:
            target_h, target_w = height, width
        image = image.astype(np.float32)
        if kwargs.get("do_rescale", self.do_rescale):
            image *= kwargs.get("rescale_factor", self.rescale_factor)
        if kwargs.get("do_normalize", self.do_normalize):
            mean = np.asarray(
                kwargs.get("image_mean", self.image_mean), dtype=np.float32
            )[:, None, None]
            std = np.asarray(kwargs.get("image_std", self.image_std), dtype=np.float32)[
                :, None, None
            ]
            image = (image - mean) / std

        patch = settings["patch_size"]
        merge = settings["merge_size"]
        temporal = settings["temporal_patch_size"]
        grid_h, grid_w = target_h // patch, target_w // patch
        patches = image.reshape(
            channels,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        ).transpose(1, 4, 2, 5, 0, 3, 6)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None],
            (*patches.shape[:5], temporal, *patches.shape[5:]),
        )
        patches = patches.reshape(grid_h * grid_w, channels * temporal * patch * patch)
        return patches, [1, grid_h, grid_w]

    def __call__(self, images, return_tensors=None, **kwargs):
        images = _flatten_images(images)
        if not images:
            raise ValueError("At least one image is required.")
        patches, grids = zip(*(self._process_one(image, **kwargs) for image in images))
        return BatchFeature(
            data={
                "pixel_values": np.concatenate(patches, axis=0),
                "image_grid_thw": np.asarray(grids, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        settings = self._settings(images_kwargs or {})
        target_h, target_w = smart_resize(
            settings["temporal_patch_size"],
            height,
            width,
            temporal_factor=settings["temporal_patch_size"],
            factor=settings["patch_size"]
            * settings["merge_size"]
            * settings["patch_expand_factor"],
            min_pixels=settings["min_image_tokens"],
            max_pixels=settings["max_image_tokens"],
        )
        return target_h // settings["patch_size"] * (target_w // settings["patch_size"])


class Glm5NextVideoProcessor(Glm5NextImageProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]
    sample_frames_in_loader = True

    def __init__(
        self,
        max_image_tokens: int = 240000,
        fps: float = 2.0,
        max_duration: int = 0,
        max_frames: Optional[int] = 2048,
        do_sample_frames: bool = True,
        num_frames: int = 16,
        max_image_size=None,
        **kwargs,
    ):
        super().__init__(max_image_tokens=max_image_tokens, **kwargs)
        self.fps = fps
        self.max_duration = max_duration
        self.max_frames = max_frames
        self.do_sample_frames = do_sample_frames
        self.num_frames = num_frames
        self.max_image_size = max_image_size or {"longest_edge": 28 * 28 * 2 * 30000}

    def sample_frames(self, metadata, fps=None, **kwargs):
        source_fps = getattr(metadata, "fps", None)
        total_frames = getattr(metadata, "total_num_frames", None)
        if source_fps is None or total_frames is None:
            raise ValueError("Video FPS and total frame count are required.")
        duration = getattr(metadata, "duration", None)
        if not duration:
            duration = round((total_frames - 1) / source_fps) + 1
        effective_duration = (
            duration if self.max_duration <= 0 else min(duration, self.max_duration)
        )
        cap = kwargs.get("max_frames", self.max_frames)
        cap = 2048 if cap is None else cap
        target_fps = self.fps if fps is None else fps
        count = min(int(effective_duration * target_fps), cap)

        timestamps = [index / source_fps for index in range(total_frames)]
        if total_frames < count:
            indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
        else:
            indices = []
            current_second = 0.0
            interval = 1 / target_fps
            for frame_index, timestamp in enumerate(timestamps):
                if timestamp >= current_second:
                    current_second += interval
                    indices.append(frame_index)
                    if current_second >= int(duration):
                        break
        if len(indices) < count:
            start = indices[0] if indices else 0
            end = indices[-1] if indices else max(total_frames - 1, 0)
            indices = np.linspace(start, end, count, dtype=int).tolist()
        elif len(indices) > count:
            indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()

        indices = np.asarray(list(dict.fromkeys(indices)), dtype=int)
        if len(indices) & 1:
            indices = np.append(indices, indices[-1])
        return indices

    def video_sampling_defaults(self):
        cap = 2048 if self.max_frames is None else self.max_frames
        return {
            "fps": self.fps,
            "min_frames": self.temporal_patch_size,
            "max_frames": cap,
            "frame_factor": self.temporal_patch_size,
        }

    @staticmethod
    def _as_videos(videos):
        if isinstance(videos, np.ndarray) and videos.ndim == 4:
            return [videos]
        if not isinstance(videos, (list, tuple)):
            return [videos]
        if not videos:
            return []
        first = videos[0]
        if getattr(first, "ndim", None) == 3 or hasattr(first, "convert"):
            return [videos]
        return list(videos)

    def _process_video(self, video, **kwargs):
        if isinstance(video, np.ndarray) and video.ndim == 4:
            frames = video
            if frames.shape[-1] in (1, 3, 4):
                frames = frames.transpose(0, 3, 1, 2)
        else:
            frames = np.stack(
                [
                    _to_numpy_processor_image(
                        frame, kwargs.get("do_convert_rgb", self.do_convert_rgb)
                    )
                    for frame in video
                ]
            )
        if kwargs.get("do_convert_rgb", self.do_convert_rgb) and frames.shape[1] == 1:
            frames = np.repeat(frames, 3, axis=1)
        elif kwargs.get("do_convert_rgb", self.do_convert_rgb) and frames.shape[1] == 4:
            frames = frames[:, :3]
        num_frames, channels, height, width = frames.shape
        settings = self._settings(kwargs)
        if kwargs.get("do_resize", self.do_resize):
            target_h, target_w, content_h, content_w = _resize_geometry(
                num_frames, height, width, **settings
            )
            frames = _resize_frames_pil(
                frames,
                content_h,
                content_w,
                kwargs.get("resample", self.resample),
            )
            frames = np.pad(
                frames,
                (
                    (0, 0),
                    (0, 0),
                    (0, target_h - content_h),
                    (0, target_w - content_w),
                ),
            )
        else:
            target_h, target_w = height, width
        frames = frames.astype(np.float32)
        if kwargs.get("do_rescale", self.do_rescale):
            frames *= kwargs.get("rescale_factor", self.rescale_factor)
        if kwargs.get("do_normalize", self.do_normalize):
            mean = np.asarray(
                kwargs.get("image_mean", self.image_mean), dtype=np.float32
            )[None, :, None, None]
            std = np.asarray(kwargs.get("image_std", self.image_std), dtype=np.float32)[
                None, :, None, None
            ]
            frames = (frames - mean) / std

        temporal = settings["temporal_patch_size"]
        if pad := -num_frames % temporal:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
            num_frames += pad
        patch, merge = settings["patch_size"], settings["merge_size"]
        grid_t = num_frames // temporal
        grid_h, grid_w = target_h // patch, target_w // patch
        patches = frames.reshape(
            grid_t,
            temporal,
            channels,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        ).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = patches.reshape(
            grid_t * grid_h * grid_w, channels * temporal * patch * patch
        )
        return patches, [grid_t, grid_h, grid_w]

    def __call__(self, videos, return_tensors=None, **kwargs):
        videos = self._as_videos(videos)
        if not videos:
            raise ValueError("At least one video is required.")
        processed = [self._process_video(video, **kwargs) for video in videos]
        patches, grids = zip(*processed)
        return BatchFeature(
            data={
                "pixel_values_videos": np.concatenate(patches, axis=0),
                "video_grid_thw": np.asarray(grids, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )

    def preprocess(self, videos, **kwargs):
        return self(videos, **kwargs)

    def get_number_of_video_patches(
        self, num_frames, height, width, videos_kwargs=None
    ):
        settings = self._settings(videos_kwargs or {})
        target_h, target_w = smart_resize(
            num_frames,
            height,
            width,
            temporal_factor=settings["temporal_patch_size"],
            factor=settings["patch_size"]
            * settings["merge_size"]
            * settings["patch_expand_factor"],
            min_pixels=settings["min_image_tokens"],
            max_pixels=settings["max_image_tokens"],
        )
        grid_t = math.ceil(num_frames / settings["temporal_patch_size"])
        return (
            grid_t
            * (target_h // settings["patch_size"])
            * (target_w // settings["patch_size"])
        )


def _load_json(model_path, filename):
    local = Path(model_path) / filename
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_path, filename)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


class Glm5NextProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image|>"
        self.video_token = getattr(tokenizer, "video_token", None) or "<|video|>"
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or (
            tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        del kwargs
        merge_length = self.image_processor.merge_size**2
        count = int(np.prod(image_inputs["image_grid_thw"][image_idx])) // merge_length
        return self.image_token * count

    def replace_video_token(self, video_inputs, video_idx, **kwargs):
        metadata = kwargs.get("video_metadata")
        if metadata is not None and not isinstance(metadata, (list, tuple)):
            metadata = [metadata]
        fps = kwargs.get("fps")
        if isinstance(fps, (list, tuple)):
            fps = fps[video_idx] if video_idx < len(fps) else None
        return self._video_replacement(
            video_inputs["video_grid_thw"][video_idx],
            metadata[video_idx] if metadata and video_idx < len(metadata) else None,
            fps,
        )

    def replace_frame_token_id(self, timestamp_sec, num_image_tokens=1):
        return (
            f"<|begin_of_image|>{self.image_token * num_image_tokens}"
            f"<|end_of_image|>{timestamp_sec:.1f} seconds"
        )

    def _video_replacement(self, grid, metadata=None, fps=None):
        grid_t, grid_h, grid_w = [int(value) for value in grid]
        per_frame = grid_h * grid_w // self.video_processor.merge_size**2
        timestamps = None
        if metadata is not None:
            timestamps = getattr(metadata, "timestamps", None)
            if timestamps is None and isinstance(metadata, dict):
                timestamps = metadata.get("timestamps")
        if timestamps is None:
            fallback_fps = fps or 24
            if fps is None:
                warnings.warn(
                    "Video timestamps were not supplied; defaulting to 24 FPS.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            timestamps = [
                index * self.video_processor.temporal_patch_size / fallback_fps
                for index in range(grid_t)
            ]
        else:
            timestamps = list(timestamps)[:: self.video_processor.temporal_patch_size]
        if not timestamps:
            timestamps = [0.0]
        timestamps = (timestamps + [timestamps[-1]] * grid_t)[:grid_t]
        return "".join(
            self.replace_frame_token_id(float(timestamp), per_frame)
            for timestamp in timestamps
        )

    def create_mm_token_type_ids(self, input_ids):
        """Match Transformers: 0=text, 1=image, and 2=video image tokens."""
        output = []
        for row in input_ids:
            if hasattr(row, "tolist"):
                row = row.tolist()
            ids = np.asarray(row)
            token_types = np.zeros_like(ids)
            starts = np.cumsum(ids == self.video_start_id)
            ends = np.cumsum(ids == self.video_end_id)
            in_video = starts > ends
            token_types[(ids == self.image_token_id) & in_video] = 2
            token_types[(ids == self.image_token_id) & ~in_video] = 1
            output.append(token_types.tolist())
        return output

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        data = {}
        if image_sizes is not None:
            merge_size = kwargs.get("merge_size", self.image_processor.merge_size)
            patches = [
                self.image_processor.get_number_of_image_patches(*size, kwargs)
                for size in image_sizes
            ]
            data["num_image_patches"] = patches
            data["num_image_tokens"] = [count // merge_size**2 for count in patches]
        if video_sizes is not None:
            merge_size = kwargs.get("merge_size", self.video_processor.merge_size)
            patches = [
                self.video_processor.get_number_of_video_patches(*size, kwargs)
                for size in video_sizes
            ]
            data["num_video_tokens"] = [count // merge_size**2 for count in patches]
        try:
            from transformers.processing_utils import MultiModalData

            return MultiModalData(**data)
        except ImportError:
            return data

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def __call__(
        self,
        images=None,
        text: Union[str, List[str], None] = None,
        videos=None,
        video_metadata=None,
        fps=None,
        **kwargs,
    ):
        text_kwargs = dict(kwargs.pop("text_kwargs", {}))
        image_kwargs = dict(kwargs.pop("images_kwargs", {}))
        video_kwargs = dict(kwargs.pop("videos_kwargs", {}))
        vision_keys = {
            "do_convert_rgb",
            "do_normalize",
            "do_resize",
            "do_rescale",
            "image_mean",
            "image_std",
            "max_image_tokens",
            "merge_size",
            "min_image_tokens",
            "patch_expand_factor",
            "patch_size",
            "resample",
            "rescale_factor",
            "size",
            "temporal_patch_size",
        }
        for name in tuple(kwargs):
            if name in vision_keys:
                value = kwargs.pop(name)
                image_kwargs.setdefault(name, value)
                video_kwargs.setdefault(name, value)

        padding = text_kwargs.pop("padding", kwargs.pop("padding", False))
        return_token_type_ids = text_kwargs.pop(
            "return_token_type_ids", kwargs.pop("return_token_type_ids", False)
        )
        return_mm_token_type_ids = text_kwargs.pop(
            "return_mm_token_type_ids",
            kwargs.pop("return_mm_token_type_ids", True),
        )
        return_tensors = text_kwargs.pop(
            "return_tensors", kwargs.pop("return_tensors", None)
        )
        video_metadata = video_kwargs.pop("video_metadata", video_metadata)
        fps = video_kwargs.pop("fps", fps)
        return_metadata = kwargs.pop("return_metadata", False)
        video_kwargs.pop("return_metadata", None)
        video_kwargs.pop("do_sample_frames", None)

        image_inputs = (
            self.image_processor(images, **image_kwargs) if images is not None else {}
        )
        video_inputs = (
            self.video_processor(videos, **video_kwargs) if videos is not None else {}
        )
        text = (
            [""] if text is None else ([text] if isinstance(text, str) else list(text))
        )

        grids = image_inputs.get("image_grid_thw")
        if grids is not None:
            image_idx = 0
            for row in range(len(text)):
                while self.image_token in text[row]:
                    if image_idx >= len(grids):
                        raise ValueError(
                            "The prompt contains more image markers than supplied images."
                        )
                    count = (
                        int(np.prod(grids[image_idx]))
                        // self.image_processor.merge_size**2
                    )
                    text[row] = text[row].replace(
                        self.image_token, "<|placeholder|>" * count, 1
                    )
                    image_idx += 1
                text[row] = text[row].replace("<|placeholder|>", self.image_token)
            if image_idx != len(grids):
                raise ValueError(
                    "The number of supplied images does not match the image markers."
                )

        video_grids = video_inputs.get("video_grid_thw")
        if video_grids is not None:
            video_idx = 0
            metadata = video_metadata or [None] * len(video_grids)
            if not isinstance(metadata, (list, tuple)):
                metadata = [metadata]
            fallback_fps = fps if isinstance(fps, (list, tuple)) else None
            for row in range(len(text)):
                while self.video_token in text[row]:
                    if video_idx >= len(video_grids):
                        raise ValueError(
                            "The prompt contains more video markers than supplied videos."
                        )
                    replacement = self._video_replacement(
                        video_grids[video_idx],
                        metadata[video_idx] if video_idx < len(metadata) else None,
                        (
                            fallback_fps[video_idx]
                            if fallback_fps is not None
                            and video_idx < len(fallback_fps)
                            else fps
                        ),
                    )
                    text[row] = text[row].replace(self.video_token, replacement, 1)
                    video_idx += 1
            if video_idx != len(video_grids):
                raise ValueError(
                    "The number of supplied videos does not match the video markers."
                )

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            return_token_type_ids=return_token_type_ids,
            return_tensors=return_tensors,
            **text_kwargs,
            **kwargs,
        )
        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(
                text_inputs["input_ids"]
            )
        if return_metadata and video_metadata is not None:
            video_inputs["video_metadata"] = video_metadata
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        names = list(getattr(self.tokenizer, "model_input_names", []))
        names += self.image_processor.model_input_names
        names += self.video_processor.model_input_names
        return list(dict.fromkeys(names + ["mm_token_type_ids"]))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        processor_kwargs = dict(kwargs)
        trust_remote_code = processor_kwargs.pop("trust_remote_code", True)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        config = _load_json(pretrained_model_name_or_path, "processor_config.json")
        image_config = dict(config.get("image_processor", {}))
        video_config = dict(config.get("video_processor", {}))
        image_config.pop("image_processor_type", None)
        video_config.pop("video_processor_type", None)
        return cls(
            image_processor=Glm5NextImageProcessor(**image_config),
            video_processor=Glm5NextVideoProcessor(**video_config),
            tokenizer=tokenizer,
        )


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
    "Glm5NextVideoProcessor",
    "smart_resize",
]


install_auto_processor_patch("glm5_next", Glm5NextProcessor)
