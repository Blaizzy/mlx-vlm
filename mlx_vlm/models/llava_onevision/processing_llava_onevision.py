"""
Processor class for LLaVA-OneVision.

The image processor that ``AutoImageProcessor`` resolves for this model type is
torchvision-backed, and the video processor requires torchvision too, so both are
bypassed here: images go through the PIL backend and video frames are handled with
numpy, keeping mlx-vlm free of a torch dependency.
"""

import json
import math
from pathlib import Path

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx

_IMAGE_MARKER = "<mlx_image_placeholder>"
_VIDEO_MARKER = "<mlx_video_placeholder>"


class LlavaOnevisionVideoProcessor:
    """Resizes and normalizes video frames to the vision tower's input size.

    Kept independent of the torchvision-backed upstream video processor so that
    mlx-vlm needs no torch install; ``LlavaOnevisionProcessor`` exposes it as its
    ``video_processor`` component, which is how callers detect native video
    support.
    """

    def __init__(
        self,
        size=(384, 384),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor: float = 1 / 255,
    ):
        self.size = tuple(size)
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.rescale_factor = rescale_factor

    def __call__(self, videos, **kwargs) -> np.ndarray:
        from PIL import Image

        if not isinstance(videos, (list, tuple)):
            videos = [videos]

        height, width = self.size
        processed = []
        for video in videos:
            frames = []
            for frame in video:
                if not isinstance(frame, Image.Image):
                    array = np.asarray(frame)
                    # load_video yields channels-first frames; PIL needs HWC.
                    if (
                        array.ndim == 3
                        and array.shape[0] in (1, 3, 4)
                        and array.shape[-1] not in (1, 3, 4)
                    ):
                        array = array.transpose(1, 2, 0)
                    frame = Image.fromarray(array.astype(np.uint8))
                frame = frame.convert("RGB").resize(
                    (width, height), Image.Resampling.BICUBIC
                )
                array = np.asarray(frame, dtype=np.float32) * self.rescale_factor
                array = (array - self.image_mean) / self.image_std
                frames.append(array.transpose(2, 0, 1))
            processed.append(np.stack(frames))
        return np.stack(processed)


class LlavaOnevisionProcessor:
    """Pairs the PIL image processor and tokenizer, expanding visual placeholders."""

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        num_image_tokens: int = 729,
        vision_aspect_ratio: str = "anyres_max_9",
        vision_feature_select_strategy: str = "full",
        image_token: str = "<image>",
        video_token: str = "<video>",
        video_size=(384, 384),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.vision_aspect_ratio = vision_aspect_ratio
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.video_token = video_token
        self.video_processor = LlavaOnevisionVideoProcessor(
            size=video_size, image_mean=image_mean, image_std=image_std
        )
        self.chat_template = getattr(tokenizer, "chat_template", None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer
        from transformers.models.llava_onevision.image_processing_pil_llava_onevision import (
            LlavaOnevisionImageProcessorPil,
        )

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        image_processor = LlavaOnevisionImageProcessorPil.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        proc_kwargs = {}
        model_dir = Path(pretrained_model_name_or_path)
        proc_cfg_path = model_dir / "processor_config.json"
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for key in (
                "num_image_tokens",
                "vision_feature_select_strategy",
                "image_token",
                "video_token",
            ):
                if key in proc_cfg:
                    proc_kwargs[key] = proc_cfg[key]

        video_cfg_path = model_dir / "video_preprocessor_config.json"
        if video_cfg_path.exists():
            with open(video_cfg_path) as f:
                video_cfg = json.load(f)
            size = video_cfg.get("size") or {}
            if "height" in size and "width" in size:
                proc_kwargs["video_size"] = (size["height"], size["width"])
            for key in ("image_mean", "image_std"):
                if key in video_cfg:
                    proc_kwargs[key] = video_cfg[key]

        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_cfg = json.load(f)
            if "vision_aspect_ratio" in model_cfg:
                proc_kwargs["vision_aspect_ratio"] = model_cfg["vision_aspect_ratio"]

        return cls(image_processor=image_processor, tokenizer=tokenizer, **proc_kwargs)

    def __call__(
        self,
        images=None,
        text: (
            TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]
        ) = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if images is None and videos is None and text is None:
            raise ValueError("You have to specify at least images, videos or text.")

        kwargs.pop("return_tensors", None)
        kwargs.pop("fps", None)
        kwargs.pop("audio", None)
        padding = kwargs.pop("padding", False)

        image_inputs = {}
        if images is not None:
            image_inputs = dict(
                self.image_processor(images, return_tensors="np", **kwargs)
            )

        video_inputs = {}
        if videos is not None:
            video_inputs = {"pixel_values_videos": self.video_processor(videos)}

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        prompt_strings = text
        if text is not None:
            # One iterator per batch, so images and videos are consumed in prompt
            # order across every sample -- matching the reference processor.
            image_sizes = iter(
                np.asarray(image_inputs["image_sizes"]).tolist() if image_inputs else ()
            )
            videos = iter(
                np.asarray(video_inputs["pixel_values_videos"]) if video_inputs else ()
            )
            tile_size = (
                np.asarray(image_inputs["pixel_values"]).shape[-2:]
                if image_inputs
                else (0, 0)
            )
            prompt_strings = [
                self._expand_placeholders(sample, image_sizes, videos, tile_size)
                for sample in text
            ]

        text_inputs = (
            self.tokenizer(prompt_strings, padding=padding, **kwargs) if text else {}
        )

        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **video_inputs})
        )

    def _expand_placeholders(self, sample: str, image_sizes, videos, tile_size) -> str:
        """Replace each visual token with as many copies as the model will emit rows."""
        height, width = tile_size
        while self.image_token in sample:
            try:
                orig_height, orig_width = next(image_sizes)
            except StopIteration:
                raise ValueError(
                    "More image placeholders than images were provided."
                ) from None
            num_image_tokens = self._get_number_of_features(
                orig_height, orig_width, height, width
            )
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1
            sample = sample.replace(
                self.image_token, _IMAGE_MARKER * num_image_tokens, 1
            )

        patches_side = int(math.sqrt(self.num_image_tokens))
        pooled_side = math.ceil(patches_side / 2)
        while self.video_token in sample:
            try:
                video = next(videos)
            except StopIteration:
                raise ValueError(
                    "More video placeholders than videos were provided."
                ) from None
            # one trailing newline token closes the whole video
            num_video_tokens = len(video) * pooled_side * pooled_side + 1
            sample = sample.replace(
                self.video_token, _VIDEO_MARKER * num_video_tokens, 1
            )

        return sample.replace(_IMAGE_MARKER, self.image_token).replace(
            _VIDEO_MARKER, self.video_token
        )

    def _get_number_of_features(
        self, orig_height: int, orig_width: int, height: int, width: int
    ) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height = height_best_resolution // height
        scale_width = width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height,
            orig_width,
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )
        return unpadded_features + newline_features + self.num_image_tokens

    def _get_unpadded_features(
        self, height, width, patches_height, patches_width, scale_height, scale_width
    ):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
        ratio = math.sqrt(
            current_height * current_width / (max_num_patches * patches_height**2)
        )
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(
                current_width // ratio
            )
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + ["pixel_values", "image_sizes", "pixel_values_videos"]
            )
        )


__all__ = ["LlavaOnevisionProcessor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("llava_onevision", LlavaOnevisionProcessor)
