import json
from pathlib import Path
from typing import Optional

import numpy as np
from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch, load_chat_template
from ..gemma4.processing_gemma4 import (
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4VideoProcessor,
    _convert_to_rgb,
    _to_channel_first,
)


def _convert_image_to_model_patches(image: np.ndarray, model_patch_size: int):
    channels, height, width = image.shape
    patch_height = height // model_patch_size
    patch_width = width // model_patch_size
    patches = image.reshape(
        channels,
        patch_height,
        model_patch_size,
        patch_width,
        model_patch_size,
    )
    patches = patches.transpose(1, 3, 2, 4, 0)
    patches = patches.reshape(
        patch_height * patch_width,
        model_patch_size * model_patch_size * channels,
    )
    grid = np.meshgrid(
        np.arange(patch_width, dtype=np.int64),
        np.arange(patch_height, dtype=np.int64),
        indexing="xy",
    )
    positions = np.stack(grid, axis=-1).reshape(-1, 2)
    return patches.astype(np.float32), positions


def _pad_patches(patches: np.ndarray, positions: np.ndarray, target_length: int):
    current_length = patches.shape[0]
    if current_length > target_length:
        return patches[:target_length], positions[:target_length]
    padding_length = target_length - current_length
    if padding_length == 0:
        return patches, positions
    patches = np.pad(
        patches,
        ((0, padding_length), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    positions = np.pad(
        positions,
        ((0, padding_length), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    return patches, positions


class Gemma4UnifiedImageProcessor(Gemma4ImageProcessor):
    model_input_names = [
        "pixel_values",
        "image_position_ids",
        "num_soft_tokens_per_image",
    ]

    def __init__(
        self,
        model_patch_size: Optional[int] = None,
        mm_posemb_size: Optional[int] = None,
        num_soft_tokens: Optional[int] = None,
        **kwargs,
    ):
        if num_soft_tokens is not None and "max_soft_tokens" not in kwargs:
            kwargs["max_soft_tokens"] = num_soft_tokens
        super().__init__(**kwargs)
        self.model_patch_size = model_patch_size or (
            self.patch_size * self.pooling_kernel_size
        )
        self.mm_posemb_size = mm_posemb_size

    def preprocess(self, images, **kwargs):
        patch_size = kwargs.get("patch_size", self.patch_size)
        max_soft_tokens = kwargs.get("max_soft_tokens", self.max_soft_tokens)
        pooling_kernel_size = kwargs.get(
            "pooling_kernel_size", self.pooling_kernel_size
        )
        model_patch_size = kwargs.get(
            "model_patch_size", patch_size * pooling_kernel_size
        )
        max_patches = max_soft_tokens * pooling_kernel_size**2

        images = self.fetch_images(images)
        if not isinstance(images, list):
            images = [images]

        from transformers.image_utils import (
            ChannelDimension,
            infer_channel_dimension_format,
            make_flat_list_of_images,
            to_numpy_array,
            valid_images,
        )

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type.")

        if self.do_convert_rgb:
            images = [_convert_to_rgb(img) for img in images]

        pixel_values = []
        image_position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            image = to_numpy_array(image)
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
            patches, positions = _convert_image_to_model_patches(
                image, model_patch_size
            )
            num_soft_tokens_per_image.append(patches.shape[0])
            patches, positions = _pad_patches(patches, positions, max_soft_tokens)
            pixel_values.append(patches)
            image_position_ids.append(positions)

        data = {
            "pixel_values": np.stack(pixel_values),
            "image_position_ids": np.stack(image_position_ids),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return data, num_soft_tokens_per_image


class Gemma4UnifiedVideoProcessor(Gemma4VideoProcessor):
    model_input_names = ["pixel_values_videos", "video_position_ids"]

    def __init__(
        self,
        model_patch_size: Optional[int] = None,
        mm_posemb_size: Optional[int] = None,
        num_soft_tokens: Optional[int] = None,
        **kwargs,
    ):
        if num_soft_tokens is not None and "max_soft_tokens" not in kwargs:
            kwargs["max_soft_tokens"] = num_soft_tokens
        self.do_resize = kwargs.pop("do_resize", True)
        super().__init__(**kwargs)
        self.model_patch_size = model_patch_size or (
            self.patch_size * self.pooling_kernel_size
        )
        self.mm_posemb_size = mm_posemb_size

    def __call__(self, videos, fps=None):
        if not isinstance(videos, list):
            videos = [videos]

        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2

        if fps is None:
            fps = [self.default_fps] * len(videos)
        elif not isinstance(fps, list):
            fps = [fps] * len(videos)

        processed = []
        processed_positions = []
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
            if self.do_resize:
                video = self._resize_frames(video, max_patches)

            video_f = video.astype(np.float32)
            if self.do_rescale and video.dtype == np.uint8:
                video_f = video_f * self.rescale_factor

            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
                std = np.array(self.image_std, dtype=np.float32)[:, None, None]
                video_f = (video_f - mean) / std

            frame_patches = []
            frame_positions = []
            tokens_per_frame = None
            for frame in video_f:
                patches, positions = _convert_image_to_model_patches(
                    frame, self.model_patch_size
                )
                if tokens_per_frame is None:
                    tokens_per_frame = patches.shape[0]
                patches, positions = _pad_patches(
                    patches, positions, self.max_soft_tokens
                )
                frame_patches.append(patches)
                frame_positions.append(positions)

            T = video_f.shape[0]
            processed.append(np.stack(frame_patches))
            processed_positions.append(np.stack(frame_positions))
            num_frames_per_video.append(T)
            num_soft_tokens_per_frame.append(int(tokens_per_frame or 0))
            sr = fps[i] if fps[i] and fps[i] > 0 else self.default_fps
            frame_timestamps.append([float(j) / float(sr) for j in range(T)])

        return {
            "pixel_values_videos": np.concatenate(processed, axis=0),
            "video_position_ids": np.concatenate(processed_positions, axis=0),
            "num_frames_per_video": num_frames_per_video,
            "num_soft_tokens_per_frame": num_soft_tokens_per_frame,
            "frame_timestamps": frame_timestamps,
        }


class Gemma4UnifiedAudioFeatureExtractor:
    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 640,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        audio_samples_per_token: int = 640,
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.audio_samples_per_token = audio_samples_per_token

    def _extract_waveform_features(self, waveform: np.ndarray):
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        pad_len = (-len(waveform)) % self.audio_samples_per_token
        if pad_len:
            waveform = np.pad(waveform, (0, pad_len), constant_values=0)
        features = waveform.reshape(-1, self.audio_samples_per_token)
        mask = np.ones(features.shape[0], dtype=bool)
        return features, mask

    def __call__(
        self,
        raw_speech,
        padding: bool | str = "longest",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [raw_speech]

        features = []
        masks = []
        for waveform in raw_speech:
            feature, mask = self._extract_waveform_features(waveform)
            if max_length is not None and truncation:
                feature = feature[:max_length]
                mask = mask[:max_length]
            features.append(feature)
            masks.append(mask)

        if max_length is None:
            max_len = max((f.shape[0] for f in features), default=0)
        else:
            max_len = max_length

        padded_features = []
        padded_masks = []
        for feature, mask in zip(features, masks):
            length = min(feature.shape[0], max_len)
            padded = np.full(
                (max_len, self.audio_samples_per_token),
                self.padding_value,
                dtype=np.float32,
            )
            padded_mask = np.zeros((max_len,), dtype=bool)
            if length:
                padded[:length] = feature[:length]
                padded_mask[:length] = mask[:length]
            padded_features.append(padded)
            padded_masks.append(padded_mask)

        return BatchFeature(
            data={
                "input_features": np.stack(padded_features),
                "input_features_mask": np.stack(padded_masks),
            }
        )


class Gemma4UnifiedProcessor(Gemma4Processor):
    model_type = "gemma4_unified"
    image_processor_class = "Gemma4UnifiedImageProcessor"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        video_processor = kwargs.pop("video_processor", None)
        if image_processor is None:
            image_processor = Gemma4UnifiedImageProcessor()
        if video_processor is None:
            video_processor = Gemma4UnifiedVideoProcessor(
                patch_size=getattr(image_processor, "patch_size", 16),
                pooling_kernel_size=getattr(image_processor, "pooling_kernel_size", 3),
                model_patch_size=getattr(image_processor, "model_patch_size", None),
                do_rescale=getattr(image_processor, "do_rescale", True),
                rescale_factor=getattr(image_processor, "rescale_factor", 1 / 255),
                do_normalize=getattr(image_processor, "do_normalize", False),
                image_mean=getattr(image_processor, "image_mean", None),
                image_std=getattr(image_processor, "image_std", None),
            )
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
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

        proc_config = {}
        if is_local:
            for name in ("processor_config.json", "preprocessor_config.json"):
                config_path = model_path / name
                if config_path.exists():
                    proc_config = json.loads(config_path.read_text())
                    break
        else:
            try:
                from huggingface_hub import hf_hub_download

                for name in ("processor_config.json", "preprocessor_config.json"):
                    try:
                        config_path = Path(
                            hf_hub_download(pretrained_model_name_or_path, name)
                        )
                        proc_config = json.loads(config_path.read_text())
                        break
                    except Exception:
                        pass
            except ImportError:
                pass

        ip_config = {}
        if isinstance(proc_config.get("image_processor"), dict):
            ip_config = dict(proc_config["image_processor"])
        ip_config.pop("image_processor_type", None)

        vp_config = {}
        if isinstance(proc_config.get("video_processor"), dict):
            vp_config = dict(proc_config["video_processor"])
        else:
            for key in (
                "patch_size",
                "pooling_kernel_size",
                "model_patch_size",
                "mm_posemb_size",
                "do_rescale",
                "rescale_factor",
                "do_normalize",
                "image_mean",
                "image_std",
            ):
                if key in ip_config:
                    vp_config[key] = ip_config[key]
        vp_config.pop("video_processor_type", None)

        fe_config = {}
        if isinstance(proc_config.get("feature_extractor"), dict):
            fe_config = dict(proc_config["feature_extractor"])
        fe_config.pop("feature_extractor_type", None)

        image_processor = Gemma4UnifiedImageProcessor(**ip_config)
        video_processor = Gemma4UnifiedVideoProcessor(**vp_config)
        feature_extractor = Gemma4UnifiedAudioFeatureExtractor(**fe_config)

        image_seq_length = proc_config.get(
            "image_seq_length", ip_config.get("max_soft_tokens", 280)
        )
        audio_seq_length = proc_config.get("audio_seq_length", 750)
        audio_ms_per_token = proc_config.get("audio_ms_per_token", 40)

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
            audio_seq_length=audio_seq_length,
            audio_ms_per_token=audio_ms_per_token,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )


__all__ = [
    "Gemma4UnifiedAudioFeatureExtractor",
    "Gemma4UnifiedImageProcessor",
    "Gemma4UnifiedVideoProcessor",
    "Gemma4UnifiedProcessor",
]


install_auto_processor_patch("gemma4_unified", Gemma4UnifiedProcessor)
