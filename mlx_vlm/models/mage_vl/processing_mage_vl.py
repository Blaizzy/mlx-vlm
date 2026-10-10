"""Torch-free processor for Mage-VL.

The checkpoint's own remote-code processor (`processing_mage_vl.MageVLProcessor` on the hub) is
torch-native — its ``__call__`` builds torch tensors, so mlx-vlm's ``prepare_inputs`` dies with
``ones_like(): argument 'input' must be Tensor, not array``. This in-repo processor follows the
``qwen3_vl`` pattern: numpy/mlx end to end, installed over AutoProcessor for ``model_type
mage_vl`` so ``mlx_vlm.load`` returns it without ``trust_remote_code``.

Frame-sampled video follows the reference's image path: timestamped image blocks,
one grid per frame, and source frame indices in ``patch_positions``. Decoding and
sampling belong to ``utils.prepare_inputs``; this processor consumes its frames
and ``VideoMetadata``. Codec-based sparse patch selection is not implemented.
"""

import re
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import make_batched_videos

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from ..qwen3_vl.processing_qwen3_vl import _flatten_images, _to_numpy_image
from .config import VisionConfig
from .mage_vl import _positions_from_grid

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
_VISUAL_PATTERN = re.compile(
    re.escape(VISION_START)
    + r"\s*"
    + re.escape(VIDEO_PAD)
    + r"\s*"
    + re.escape(VISION_END)
    + "|"
    + re.escape(VIDEO_PAD)
    + "|"
    + re.escape(IMAGE_PAD)
)


class MageVLVideoProcessor(BaseVideoProcessor):
    """Prepare decoded frames and their times for the shared image processor."""

    def __init__(self, max_frames=32):
        self.max_frames = max_frames

    def video_sampling_defaults(self):
        # One frame per patch; keep a bounded default for dense frame attention.
        return {"min_frames": 1, "max_frames": self.max_frames, "frame_factor": 1}

    def __call__(self, videos, video_metadata=None, fps=None):
        videos = make_batched_videos(videos)
        metadata = (
            video_metadata if video_metadata is not None else [None] * len(videos)
        )
        if len(metadata) != len(videos):
            raise ValueError("Expected one video_metadata entry per video.")
        rates = fps if isinstance(fps, (list, tuple)) else [fps] * len(videos)
        if len(rates) != len(videos):
            raise ValueError("Expected one fps value per video.")

        clips = []
        for video, meta, rate in zip(videos, metadata, rates):
            if isinstance(video, str):
                raise TypeError(
                    "Use prepare_inputs() to decode and sample video paths."
                )
            frames = [_to_numpy_image(frame) for frame in video]
            if not frames:
                raise ValueError("Video input contains no frames.")
            indices = list(range(len(frames)))
            if meta is not None:
                if isinstance(meta, dict):
                    indices, rate = meta["frames_indices"], meta["fps"]
                else:
                    indices, rate = meta.frames_indices, meta.fps
            rate = 1.0 if rate is None else rate
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("Video fps must be positive and finite.")
            if len(indices) != len(frames):
                raise ValueError(
                    "Video frame indices must match the decoded frame count."
                )
            clips.append((frames, indices, [i / rate for i in indices]))
        return clips


class MageVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    # Match Qwen3VLProcessor's numpy components in torch-free installations.
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
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        if video_processor is None:
            video_processor = MageVLVideoProcessor()
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.spatial_merge_size = getattr(image_processor, "merge_size", 2)
        self.image_token = IMAGE_PAD
        self.video_token = VIDEO_PAD

    # The subset of preprocessor_config.json Qwen3VLImageProcessor consumes.
    _IMAGE_PROCESSOR_KEYS = (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "min_pixels",
        "max_pixels",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "do_convert_rgb",
    )

    @staticmethod
    def _load_preprocessor_config(pretrained_model_name_or_path) -> dict:
        import json
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            cfg = path / "preprocessor_config.json"
            return json.loads(cfg.read_text()) if cfg.exists() else {}
        try:
            from huggingface_hub import hf_hub_download

            return json.loads(
                Path(
                    hf_hub_download(
                        str(pretrained_model_name_or_path), "preprocessor_config.json"
                    )
                ).read_text()
            )
        except Exception:
            return {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        from ..qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        # trust_remote_code=False EXPLICITLY: the checkpoint's configs carry auto_map entries,
        # and leaving the argument unset makes transformers prompt interactively.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Construct the image processor DIRECTLY — no AutoImageProcessor. On a torch-free
        # install (the standard MLX setup) AutoImageProcessor resolves to a torchvision-backed
        # class and raises ImportError, which the AutoProcessor patch dispatcher swallows,
        # silently falling back to the checkpoint's torch remote-code processor — the exact
        # failure this class exists to prevent. Reported by @ismaelvega on PR #1745 and
        # mlx-community/Mage-VL-8bit discussion #1. Qwen3VLImageProcessor is the in-repo
        # numpy/PIL port with the same schema (Mage-VL's config: patch 16, merge 2,
        # temporal_patch_size 1, CLIP statistics).
        config = cls._load_preprocessor_config(pretrained_model_name_or_path)
        image_processor = Qwen3VLImageProcessor(
            **{k: config[k] for k in cls._IMAGE_PROCESSOR_KEYS if k in config}
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput]] = None,
        images=None,
        videos=None,
        video_metadata=None,
        fps=None,
        audio=None,
        min_pixels=None,
        max_pixels=None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(text, str):
            text = [text]
        text = list(text or [])
        if videos is not None and self.image_processor.temporal_patch_size != 1:
            raise ValueError("Mage-VL video input requires temporal_patch_size=1.")

        image_inputs = {}
        if images is not None or videos is not None:
            stills = iter(_flatten_images(images) if images is not None else [])
            clips = iter(
                self.video_processor(videos, video_metadata, fps)
                if videos is not None
                else []
            )
            ordered_frames, frame_indices = [], []

            def replace_visual(match):
                if match.group() == IMAGE_PAD:
                    image = next(stills, None)
                    if image is None:
                        raise ValueError("More image placeholders than images.")
                    ordered_frames.append(image)
                    frame_indices.append(0)
                    return IMAGE_PAD
                clip = next(clips, None)
                if clip is None:
                    raise ValueError("More video placeholders than videos.")
                frames, indices, seconds = clip
                ordered_frames.extend(frames)
                frame_indices.extend(indices)
                return "".join(
                    f"<{sec:.1f} seconds>{VISION_START}{IMAGE_PAD}{VISION_END}"
                    for sec in seconds
                )

            text = [_VISUAL_PATTERN.sub(replace_visual, s) for s in text]
            if next(stills, None) is not None or next(clips, None) is not None:
                raise ValueError(
                    "Each image and video must have a placeholder in text."
                )

            if ordered_frames:
                image_inputs = dict(
                    self.image_processor(
                        images=ordered_frames,
                        return_tensors="np",
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                )
                grids = np.asarray(image_inputs["image_grid_thw"])
                counts = iter(
                    (np.prod(grids, axis=1) // self.spatial_merge_size**2).tolist()
                )
                text = [
                    re.sub(re.escape(IMAGE_PAD), lambda _: IMAGE_PAD * next(counts), s)
                    for s in text
                ]

                if videos is not None:
                    # Grids stay per-frame for attention; RoPE keeps source times.
                    config = VisionConfig(spatial_merge_size=self.spatial_merge_size)
                    positions = _positions_from_grid(grids.tolist(), config)
                    positions[:, 0] = mx.array(
                        np.repeat(frame_indices, np.prod(grids, axis=1))
                    )
                    image_inputs["patch_positions"] = positions

        # mlx-vlm's process_inputs passes padding in kwargs; hardcoding it too collides.
        padding = kwargs.pop("padding", True)
        text_inputs = self.tokenizer(
            text, return_tensors="np", padding=padding, **kwargs
        )
        return BatchFeature(data=to_mlx({**dict(text_inputs), **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "patch_positions",
        ]


__all__ = ["MageVLProcessor"]

install_auto_processor_patch("mage_vl", MageVLProcessor)
