"""
Processor class for Gemma4.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/processing_gemma4.py
"""

import math
import re
from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import (
    BaseImageProcessor as HFBaseImageProcessor,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def _convert_to_rgb(image):
    from PIL import Image

    if not isinstance(image, Image.Image):
        return image
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _to_channel_first(image, input_format):
    if input_format == ChannelDimension.FIRST:
        return image
    if input_format == ChannelDimension.LAST:
        return np.transpose(image, (2, 0, 1))
    return image


class Gemma4ImageProcessor(HFBaseImageProcessor):
    """Image processor for Gemma 4.

    Aspect-ratio preserving resize, rescale to [0,1], output as channels-first.
    Patchification is handled by the model, not the image processor.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = False,
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size or {"height": 224, "width": 224}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def aspect_ratio_preserving_resize(
        self, image, patch_size, max_patches, pooling_kernel_size, input_data_format
    ):
        """Resize image preserving aspect ratio so it fits within the patch budget.

        Target dimensions are the largest that:
        1) Produce at most `max_patches` patches when patchified with `patch_size`
        2) Have height and width divisible by `pooling_kernel_size * patch_size`
        """
        if input_data_format == ChannelDimension.FIRST:
            height, width = image.shape[1], image.shape[2]
        else:
            height, width = image.shape[0], image.shape[1]

        target_px = max_patches * (patch_size**2)
        factor = math.sqrt(target_px / (height * width))
        side_mult = pooling_kernel_size * patch_size

        target_height = int(math.floor(factor * height / side_mult)) * side_mult
        target_width = int(math.floor(factor * width / side_mult)) * side_mult

        if target_height == 0 and target_width == 0:
            raise ValueError("Attempting to resize to a 0 x 0 image.")

        max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
        if target_height == 0:
            target_height = side_mult
            target_width = min(
                int(math.floor(width / height)) * side_mult, max_side_length
            )
        elif target_width == 0:
            target_width = side_mult
            target_height = min(
                int(math.floor(height / width)) * side_mult, max_side_length
            )

        if target_height == height and target_width == width:
            return image

        from PIL import Image

        if input_data_format == ChannelDimension.FIRST:
            img_arr = np.transpose(image, (1, 2, 0))
        else:
            img_arr = image

        if img_arr.dtype in (np.float32, np.float64):
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_arr)
        pil_img = pil_img.resize((target_width, target_height), resample=Image.BICUBIC)
        result = np.array(pil_img)

        if input_data_format == ChannelDimension.FIRST:
            result = np.transpose(result, (2, 0, 1))

        return result

    def preprocess(self, images: ImageInput, **kwargs):
        patch_size = kwargs.get("patch_size", self.patch_size)
        max_soft_tokens = kwargs.get("max_soft_tokens", self.max_soft_tokens)
        pooling_kernel_size = kwargs.get(
            "pooling_kernel_size", self.pooling_kernel_size
        )
        max_patches = max_soft_tokens * pooling_kernel_size**2

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image type.")

        if self.do_convert_rgb:
            images = [_convert_to_rgb(img) for img in images]

        images = [to_numpy_array(img) for img in images]

        processed = []
        num_soft_tokens_per_image = []

        for image in images:
            input_data_format = infer_channel_dimension_format(image)

            if self.do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image,
                    patch_size,
                    max_patches,
                    pooling_kernel_size,
                    input_data_format,
                )

            if self.do_rescale:
                image = image.astype(np.float32) * self.rescale_factor

            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)
                std = np.array(self.image_std, dtype=np.float32)
                if input_data_format == ChannelDimension.LAST:
                    image = (image - mean) / std
                else:
                    image = (image - mean[:, None, None]) / std[:, None, None]

            image = _to_channel_first(image, input_data_format)
            processed.append(image)

            h, w = image.shape[-2], image.shape[-1]
            num_patches = (h // patch_size) * (w // patch_size)
            num_soft_tokens_per_image.append(num_patches // (pooling_kernel_size**2))

        # Stack if all same shape, otherwise return as list for per-image processing
        shapes = {img.shape for img in processed}
        if len(shapes) == 1:
            data = {"pixel_values": np.stack(processed)}
        else:
            data = {"pixel_values": processed}

        return data, num_soft_tokens_per_image

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)


class Gemma4VideoProcessor:
    """Video processor for Gemma 4.

    Samples frames, applies the same aspect-ratio preserving resize as images
    (with a smaller per-frame token budget), rescales to [0, 1], and returns
    channel-first pixel tensors stacked across frames. The existing
    ``vision_tower`` internally patchifies each frame, so we output regular
    (N_frames, C, H, W) tensors rather than pre-patched ones.
    """

    model_input_names = ["pixel_values_videos"]

    def __init__(
        self,
        patch_size: int = 16,
        max_soft_tokens: int = 70,
        pooling_kernel_size: int = 3,
        num_frames: int = 32,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = False,
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        default_fps: float = 2.0,
    ):
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.num_frames = num_frames
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.0, 0.0, 0.0]
        self.image_std = image_std or [1.0, 1.0, 1.0]
        self.default_fps = default_fps

    def _sample_frames(self, video: np.ndarray, num_frames: int) -> np.ndarray:
        """Uniformly sample ``num_frames`` frames from ``video`` (T, C, H, W)."""
        T = video.shape[0]
        if T <= num_frames:
            return video
        idx = np.linspace(0, T - 1, num=num_frames).round().astype(np.int64)
        return video[idx]

    def _resize_frames(
        self, video: np.ndarray, max_patches: int
    ) -> np.ndarray:
        from PIL import Image

        T, C, H, W = video.shape
        target_px = max_patches * (self.patch_size**2)
        factor = math.sqrt(target_px / (H * W))
        side_mult = self.pooling_kernel_size * self.patch_size
        target_h = int(math.floor(factor * H / side_mult)) * side_mult
        target_w = int(math.floor(factor * W / side_mult)) * side_mult
        max_side = (max_patches // self.pooling_kernel_size**2) * side_mult
        if target_h == 0:
            target_h = side_mult
            target_w = min(int(math.floor(W / H)) * side_mult, max_side)
        if target_w == 0:
            target_w = side_mult
            target_h = min(int(math.floor(H / W)) * side_mult, max_side)

        if target_h == H and target_w == W:
            return video

        resized = np.empty((T, C, target_h, target_w), dtype=video.dtype)
        for i, frame in enumerate(video):
            arr = np.transpose(frame, (1, 2, 0))
            if arr.dtype in (np.float32, np.float64):
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            pil = pil.resize((target_w, target_h), resample=Image.BICUBIC)
            resized[i] = np.transpose(np.array(pil), (2, 0, 1))
        return resized

    def __call__(self, videos, fps=None):
        """Process a list of videos.

        Args:
            videos: list of numpy arrays, each shape (T, C, H, W) uint8 or float.
            fps: optional list of sampling fps per video used to compute
                per-frame timestamps. If None, uses ``default_fps`` for all.

        Returns:
            dict with:
              - pixel_values_videos: np.ndarray (N_total_frames, C, H, W)
                where all frames share the same H/W (one video at a time
                preserves sizes; cross-video sizes may differ so we return a
                list in that case)
              - num_frames_per_video: list[int]
              - num_soft_tokens_per_frame: list[int] (one per video)
              - frame_timestamps: list[list[float]] seconds per frame
        """
        if not isinstance(videos, list):
            videos = [videos]

        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2

        if fps is None:
            fps = [self.default_fps] * len(videos)
        elif not isinstance(fps, list):
            fps = [fps] * len(videos)

        processed = []
        num_frames_per_video = []
        num_soft_tokens_per_frame = []
        frame_timestamps = []

        for i, video in enumerate(videos):
            if not isinstance(video, np.ndarray):
                video = np.asarray(video)
            if video.ndim != 4:
                raise ValueError(
                    f"Expected video as (T, C, H, W), got shape {video.shape}."
                )

            video = self._sample_frames(video, self.num_frames)
            video = self._resize_frames(video, max_patches)

            video_f = video.astype(np.float32)
            if self.do_rescale and video.dtype == np.uint8:
                video_f = video_f * self.rescale_factor

            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
                std = np.array(self.image_std, dtype=np.float32)[:, None, None]
                video_f = (video_f - mean) / std

            T, _, H, W = video_f.shape
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            tokens_per_frame = num_patches // (self.pooling_kernel_size**2)

            processed.append(video_f)
            num_frames_per_video.append(T)
            num_soft_tokens_per_frame.append(int(tokens_per_frame))
            sr = fps[i] if fps[i] and fps[i] > 0 else self.default_fps
            frame_timestamps.append([float(j) / float(sr) for j in range(T)])

        shapes = {v.shape[1:] for v in processed}
        if len(shapes) == 1:
            pixel_values_videos = np.concatenate(processed, axis=0)
        else:
            pixel_values_videos = processed

        return {
            "pixel_values_videos": pixel_values_videos,
            "num_frames_per_video": num_frames_per_video,
            "num_soft_tokens_per_frame": num_soft_tokens_per_frame,
            "frame_timestamps": frame_timestamps,
        }


class Gemma4Processor(ProcessorMixin):
    """Combined processor for Gemma 4 (image + text + audio + video)."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Gemma4ImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template", "image_seq_length", "audio_seq_length"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        **kwargs,
    ):
        feature_extractor = kwargs.pop("feature_extractor", None)
        video_processor = kwargs.pop("video_processor", None)

        if image_processor is None:
            image_processor = Gemma4ImageProcessor()
        if video_processor is None:
            video_processor = Gemma4VideoProcessor()

        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length
        self.audio_ms_per_token = kwargs.pop("audio_ms_per_token", 40)

        # Image token attributes
        self.image_token_id = getattr(tokenizer, "image_token_id", None)
        self.boi_token = getattr(tokenizer, "boi_token", "")
        self.eoi_token = getattr(tokenizer, "eoi_token", "")
        self.image_token = getattr(tokenizer, "image_token", "")

        # Audio token attributes
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        self.audio_token = getattr(tokenizer, "audio_token", "")
        self.boa_token = getattr(tokenizer, "boa_token", "")
        self.eoa_token = getattr(tokenizer, "eoa_token", "")

        self.video_token = getattr(tokenizer, "video_token", "<|video|>")
        if tokenizer is not None:
            existing = tokenizer.convert_tokens_to_ids(self.video_token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if existing is None or existing == unk_id:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [self.video_token]}
                )
            self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        else:
            self.video_token_id = None

        # Precompute fallback full sequences
        image_tokens_expanded = self.image_token * image_seq_length
        self.full_image_sequence = (
            f"{self.boi_token}{image_tokens_expanded}{self.eoi_token}"
        )

        if self.audio_token and self.boa_token and self.eoa_token:
            audio_tokens_expanded = self.audio_token * audio_seq_length
            self.full_audio_sequence = (
                f"{self.boa_token}{audio_tokens_expanded}{self.eoa_token}"
            )
        else:
            self.full_audio_sequence = None

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

        self.feature_extractor = feature_extractor
        self.video_processor = video_processor

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate: int) -> int:
        """Compute number of audio soft tokens from waveform duration.

        Uses ceil(audio_duration_ms / audio_ms_per_token) capped at audio_seq_length.
        """
        num_samples = len(audio_waveform)
        duration_ms = num_samples / sampling_rate * 1000.0
        num_tokens = math.ceil(duration_ms / self.audio_ms_per_token)
        return min(num_tokens, self.audio_seq_length)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        audio: Optional[List] = None,
        videos: Optional[List] = None,
        fps: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None and audio is None and videos is None:
            raise ValueError(
                "Provide at least one of `text`, `images`, `audio`, or `videos`."
            )

        fps_kwarg = fps

        # Pop return_tensors - we handle conversion ourselves via to_mlx()
        kwargs.pop("return_tensors", None)

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise TypeError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        # ── Process images ──────────────────────────────────────────────
        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            image_data, num_soft_tokens = self.image_processor(images)
            image_inputs = image_data

            if text is not None and num_soft_tokens is not None:
                # Expand each image_token placeholder to the per-image soft token count.
                # re.sub never re-scans replaced text, so it is safe even though the
                # replacement strings themselves contain image_token.
                replacements = [
                    f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
                    for n in num_soft_tokens
                ]
                replacements_iter = iter(replacements)
                pattern = re.escape(self.image_token)
                text = [
                    re.sub(pattern, lambda _: next(replacements_iter), prompt)
                    for prompt in text
                ]
            elif text is not None:
                # Fallback: use fixed image_seq_length
                text = [
                    prompt.replace(self.image_token, self.full_image_sequence)
                    for prompt in text
                ]

        video_inputs = {}
        if videos is not None:
            video_data = self.video_processor(videos, fps=fps_kwarg)
            num_tokens_per_frame = video_data["num_soft_tokens_per_frame"]
            num_frames_per_video = video_data["num_frames_per_video"]
            frame_timestamps = video_data["frame_timestamps"]

            if text is not None:
                # Expand each <|video|> placeholder into a per-frame sequence:
                #   "<mm:ss> {boi}{video_token * n}{eoi}"  repeated per frame,
                # joined with a space, matching HF Gemma 4 processor behavior.
                replacements = []
                for n_tokens, timestamps in zip(num_tokens_per_frame, frame_timestamps):
                    frames = []
                    for secs in timestamps:
                        mm = int(secs // 60)
                        ss = int(secs % 60)
                        ts = f"{mm:02d}:{ss:02d}"
                        frames.append(
                            f"{ts} {self.boi_token}"
                            f"{self.video_token * n_tokens}"
                            f"{self.eoi_token}"
                        )
                    replacements.append(" ".join(frames))

                replacements_iter = iter(replacements)
                video_pattern = re.escape(self.video_token)
                text = [
                    re.sub(
                        video_pattern, lambda _: next(replacements_iter), prompt
                    )
                    for prompt in text
                ]

            pvv = video_data["pixel_values_videos"]
            video_inputs["pixel_values_videos"] = pvv
            video_inputs["num_frames_per_video"] = num_frames_per_video

        # ── Process audio ───────────────────────────────────────────────
        audio_inputs = {}
        if audio is not None and self.feature_extractor is not None:
            audio_arrays = []
            sampling_rates = []
            for a in audio:
                if isinstance(a, tuple):
                    audio_arrays.append(a[0])
                    sampling_rates.append(a[1])
                else:
                    audio_arrays.append(a)
                    sampling_rates.append(self.feature_extractor.sampling_rate)

            # Expand audio tokens in text based on waveform duration
            if text is not None and self.audio_token:
                num_audio_tokens = [
                    self._compute_audio_num_tokens(a, sr)
                    for a, sr in zip(audio_arrays, sampling_rates)
                ]
                replacements = [
                    (self.boa_token + self.audio_token * n + self.eoa_token)
                    for n in num_audio_tokens
                ]
                replacements_iter = iter(replacements)
                audio_pattern = re.escape(self.audio_token)
                text = [
                    re.sub(audio_pattern, lambda _: next(replacements_iter), prompt)
                    for prompt in text
                ]

            result = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_attention_mask=True,
            )
            audio_inputs["input_features"] = result["input_features"]
            if "input_features_mask" in result:
                audio_inputs["input_features_mask"] = result["input_features_mask"]

        elif (
            audio is not None
            and text is not None
            and self.audio_token
            and self.full_audio_sequence
        ):
            # No feature extractor available - use fixed audio_seq_length
            text = [
                prompt.replace(self.audio_token, self.full_audio_sequence)
                for prompt in text
            ]

        # ── Tokenize text ───────────────────────────────────────────────
        # Pop return_mm_token_type_ids before passing remaining kwargs to tokenizer
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **kwargs)

        # Generate multimodal token type IDs
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            if self.image_token_id is not None:
                mm_token_type_ids[array_ids == self.image_token_id] = 1
            if self.audio_token_id is not None:
                mm_token_type_ids[array_ids == self.audio_token_id] = 2
            if self.video_token_id is not None:
                mm_token_type_ids[array_ids == self.video_token_id] = 3
            text_inputs["token_type_ids"] = mm_token_type_ids.tolist()

        # num_frames_per_video is Python-side metadata; keep it out of to_mlx
        video_meta = {}
        if "num_frames_per_video" in video_inputs:
            video_meta["num_frames_per_video"] = video_inputs.pop(
                "num_frames_per_video"
            )

        # Merge all inputs and convert to MLX arrays
        merged = to_mlx(
            {**text_inputs, **image_inputs, **audio_inputs, **video_inputs}
        )
        merged.update(video_meta)
        return BatchFeature(data=merged)

    def save_pretrained(self, save_directory, **kwargs):
        import json
        from pathlib import Path

        super().save_pretrained(save_directory, **kwargs)

        config_path = Path(save_directory) / "processor_config.json"
        if config_path.exists() and self.feature_extractor is not None:
            config = json.loads(config_path.read_text())
            if "feature_extractor" not in config:
                fe = self.feature_extractor
                config["feature_extractor"] = {
                    "feature_extractor_type": type(fe).__name__,
                    "sampling_rate": getattr(fe, "sampling_rate", 16000),
                    "num_mel_filters": getattr(fe, "num_mel_filters", 128),
                    "fft_length": getattr(fe, "fft_length", 512),
                    "hop_length": getattr(fe, "hop_length", 160),
                    "chunk_duration": getattr(fe, "chunk_duration", 8.0),
                    "overlap_duration": getattr(fe, "overlap_duration", 1.0),
                }
                config.setdefault("audio_ms_per_token", self.audio_ms_per_token)
                config_path.write_text(json.dumps(config, indent=2))

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        kwargs.setdefault("tokenize", False)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        all_names = list(tokenizer_input_names + image_processor_input_names)
        return list(dict.fromkeys(all_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoTokenizer

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Load processor config (contains image_processor and feature_extractor settings)
        proc_config = {}
        ip_config = {}
        fe_config = {}

        def _load_json(path):
            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return None

        if is_local:
            # Try processor_config.json first, then preprocessor_config.json
            cfg = _load_json(model_path / "processor_config.json")
            if cfg is None:
                cfg = _load_json(model_path / "preprocessor_config.json")
            if cfg is not None:
                proc_config = cfg
        else:
            try:
                from huggingface_hub import hf_hub_download

                try:
                    config_path = Path(
                        hf_hub_download(
                            pretrained_model_name_or_path, "processor_config.json"
                        )
                    )
                    proc_config = json.loads(config_path.read_text())
                except Exception:
                    try:
                        config_path = Path(
                            hf_hub_download(
                                pretrained_model_name_or_path,
                                "preprocessor_config.json",
                            )
                        )
                        proc_config = json.loads(config_path.read_text())
                    except Exception:
                        pass
            except ImportError:
                pass

        # Extract image processor config
        if "image_processor" in proc_config and isinstance(
            proc_config["image_processor"], dict
        ):
            ip_config = proc_config["image_processor"]
            ip_config.pop("image_processor_type", None)
        elif "image_processor_type" not in proc_config and any(
            k in proc_config
            for k in ("patch_size", "max_soft_tokens", "pooling_kernel_size")
        ):
            # Config is flat (preprocessor_config.json format)
            ip_config = {
                k: proc_config[k]
                for k in (
                    "patch_size",
                    "max_soft_tokens",
                    "pooling_kernel_size",
                    "do_resize",
                    "do_rescale",
                    "do_normalize",
                    "rescale_factor",
                    "image_mean",
                    "image_std",
                    "size",
                )
                if k in proc_config
            }

        # Extract feature extractor config
        if "feature_extractor" in proc_config and isinstance(
            proc_config["feature_extractor"], dict
        ):
            fe_config = proc_config["feature_extractor"]
            fe_config.pop("feature_extractor_type", None)

        image_processor = Gemma4ImageProcessor(**ip_config)

        # Load audio feature extractor.
        # The standard HF checkpoint does not include a "feature_extractor" key
        # in processor_config.json, so we instantiate with defaults when the
        # config is missing — the USM parameters are fixed for all Gemma 4 models.
        feature_extractor = None
        try:
            from .audio_feature_extractor import Gemma4AudioFeatureExtractor

            feature_extractor = Gemma4AudioFeatureExtractor(**(fe_config or {}))
        except ImportError:
            try:
                from transformers import Gemma4AudioFeatureExtractor

                feature_extractor = Gemma4AudioFeatureExtractor(**(fe_config or {}))
            except (ImportError, Exception):
                pass

        image_seq_length = ip_config.get("max_soft_tokens", 280)
        audio_seq_length = proc_config.get("audio_seq_length", 750)
        audio_ms_per_token = proc_config.get("audio_ms_per_token", 40)

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
            audio_seq_length=audio_seq_length,
            audio_ms_per_token=audio_ms_per_token,
            feature_extractor=feature_extractor,
            chat_template=tokenizer.chat_template,
        )


__all__ = ["Gemma4ImageProcessor", "Gemma4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("gemma4", Gemma4Processor)
