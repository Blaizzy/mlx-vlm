import json
from dataclasses import fields
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from ..muse_glimmer.processing_muse_glimmer import _extract_final_channel
from .config import ModelConfig
from .image_processing import IMAGE_PATCH, ImageProcessor, image_tokens
from .video_processing import MuseSparkVideoProcessor


class MuseSparkFeatureExtractor(SequenceFeatureExtractor):
    model_input_names: ClassVar = ["input_values", "input_values_mask"]

    def __init__(self, sampling_rate=24000, samples_per_token=1920, **kwargs):
        kwargs.pop("feature_size", None)
        kwargs.pop("padding_side", None)
        kwargs.pop("padding_value", None)
        kwargs.pop("return_attention_mask", None)
        super().__init__(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=0.0,
            padding_side="right",
            return_attention_mask=True,
            **kwargs,
        )
        self.samples_per_token = samples_per_token

    def __call__(self, raw_speech, sampling_rate=None, return_tensors=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected audio sampled at {self.sampling_rate} Hz.")
        if isinstance(raw_speech, np.ndarray):
            raw_speech = [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        elif len(raw_speech) and np.ndim(raw_speech[0]) == 0:
            raw_speech = [raw_speech]
        waves = [np.asarray(wave, dtype=np.float32) for wave in raw_speech]
        if not waves or any(wave.ndim != 1 or wave.size == 0 for wave in waves):
            raise ValueError("Audio must contain nonempty mono waveforms.")
        lengths = [len(wave) for wave in waves]
        samples = self.samples_per_token
        padded_length = (max(lengths) + samples - 1) // samples * samples
        values = np.zeros((len(waves), padded_length), dtype=np.float32)
        mask = np.zeros(values.shape, dtype=np.int32)
        for idx, wave in enumerate(waves):
            values[idx, : len(wave)] = wave
            mask[idx, : len(wave)] = 1
        data = {
            "input_values": values,
            "input_values_mask": mask,
            "audio_lengths": lengths,
        }
        if return_tensors == "mlx":
            data = to_mlx(data)
        return BatchFeature(
            data, tensor_type=None if return_tensors == "mlx" else return_tensors
        )


class MuseSparkProcessor(ProcessorMixin):
    attributes: ClassVar = [
        "image_processor",
        "tokenizer",
        "feature_extractor",
        "video_processor",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "AutoFeatureExtractor"
    video_processor_class = "AutoVideoProcessor"
    supports_multiple_audio = True

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        feature_extractor=None,
        video_processor=None,
        chat_template=None,
        config=None,
        **kwargs,
    ):
        if video_processor is None and image_processor is not None:
            vision = getattr(config, "vision_config", None)
            video_processor = MuseSparkVideoProcessor(
                patch_size=image_processor.patch_size,
                temporal_patch_size=getattr(vision, "patch_temporal", 2),
                merge_size=image_processor.downsample_factor,
                image_mean=image_processor.image_mean,
                image_std=image_processor.image_std,
            )
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.config = config
        self.image_token = IMAGE_PATCH
        self.audio_token = "<|audio|>"
        self.video_token = "<|video|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        # <|eom|> separates reasoning/answer messages; it must not stop generation.
        self.additional_eos_token_ids = (200001, 200008)

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        videos=None,
        video_metadata=None,
        fps=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        return_tensors="mlx",
        sampling_rate=None,
        **kwargs,
    ):
        prompts = [text] if isinstance(text, str) else list(text or [])
        if not prompts:
            raise ValueError("Text prompts are required.")
        data = {}
        if images is not None:
            if self.image_processor is None:
                raise ValueError("This processor has no image encoder configuration.")
            images = images if isinstance(images, (list, tuple)) else [images]
            pixels, counts = [], []
            for image in images:
                if isinstance(image, Image.Image):
                    image = np.asarray(image.convert("RGB"))
                pixels_one, count = self.image_processor.preprocess_image(
                    np.asarray(image)
                )
                pixels.append(pixels_one)
                counts.append(count)
            prompts = self._expand(
                prompts, self.image_token, [image_tokens(n) for n in counts]
            )
            # Images may have different sizes; preserve a list of MLX arrays.
            data["pixel_values"] = pixels
        if videos is not None:
            if self.video_processor is None:
                raise ValueError("This processor has no video encoder configuration.")
            video_inputs = self.video_processor(
                videos, video_metadata=video_metadata, fps=fps
            )
            replacements = []
            for grid, timestamps in zip(
                video_inputs["video_grid_thw"], video_inputs.pop("timestamps")
            ):
                count = int(grid[1] * grid[2] // self.video_processor.merge_size**2)
                groups = [
                    self.video_processor.frame_timestamp_format.format(
                        timestamp=timestamp
                    )
                    + self.video_token * count
                    for timestamp in timestamps
                ]
                replacements.append(
                    "<|vid_start|>"
                    + "<|vid_frame_separator|>".join(groups)
                    + "<|vid_end|>"
                )
            prompts = self._expand(prompts, self.video_token, replacements)
            data.update(video_inputs)
        if audio is not None:
            if self.feature_extractor is None:
                raise ValueError("This processor has no audio encoder configuration.")
            audio_inputs = self.feature_extractor(audio, sampling_rate=sampling_rate)
            samples = self.feature_extractor.samples_per_token
            replacements = [
                "<|soft_audio_start|>"
                + self.audio_token * ((n + samples - 1) // samples)
                + "<|soft_audio_end|>"
                for n in audio_inputs["audio_lengths"]
            ]
            prompts = self._expand(prompts, self.audio_token, replacements)
            data.update(audio_inputs)
        tokenized = self.tokenizer(
            prompts,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        data.update(tokenized)
        if return_tensors == "mlx":
            # to_mlx cannot stack variable-sized images.
            pixels = data.pop("pixel_values", None)
            data = to_mlx(data)
            if pixels is not None:
                data["pixel_values"] = pixels
            return BatchFeature(data)
        return BatchFeature(data, tensor_type=return_tensors)

    @staticmethod
    def _expand(prompts, marker, replacements):
        count = sum(prompt.count(marker) for prompt in prompts)
        if count != len(replacements):
            raise ValueError(
                f"The text has {count} {marker} placeholders for {len(replacements)} inputs."
            )
        replacements = iter(replacements)
        expanded = []
        for prompt in prompts:
            parts = prompt.split(marker)
            expanded.append(
                parts[0] + "".join(next(replacements) + part for part in parts[1:])
            )
        return expanded

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def clean_output(self, text):
        return _extract_final_channel(text)

    def to_dict(self):
        data = super().to_dict()
        # The model config is saved separately as config.json.
        data.pop("config", None)
        return data

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "video_grid_thw",
            "input_values",
            "input_values_mask",
            "audio_lengths",
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer, PreTrainedConfig

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        config = kwargs.pop("config", None)
        if config is None:
            config, _ = PreTrainedConfig.get_config_dict(
                pretrained_model_name_or_path, **kwargs
            )
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        processor_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        settings = (
            json.loads(processor_path.read_text()) if processor_path.exists() else {}
        )
        image_processor = video_processor = None
        if config.vision_config is not None:
            vision = config.vision_config
            image = settings.get("image_processor", {})
            image_processor = ImageProcessor(
                patch_size=vision.patch_size,
                downsample_factor=vision.merge_size,
                min_num_image_tokens=image.get(
                    "min_image_tokens", image.get("min_num_image_tokens", 1)
                ),
                max_num_image_tokens=image.get(
                    "max_image_tokens", image.get("max_num_image_tokens", 4096)
                ),
                image_mean=tuple(image.get("image_mean", (0.5, 0.5, 0.5))),
                image_std=tuple(image.get("image_std", (0.5, 0.5, 0.5))),
            )
            video_config = settings.get("video_processor", {})
            video_args = {
                field.name: video_config[field.name]
                for field in fields(MuseSparkVideoProcessor)
                if field.name in video_config
            }
            video_args.update(
                patch_size=vision.patch_size,
                temporal_patch_size=vision.patch_temporal,
                merge_size=vision.merge_size,
            )
            video_processor = MuseSparkVideoProcessor(**video_args)
        feature_extractor = None
        if config.audio_config is not None:
            audio = settings.get("feature_extractor", {})
            feature_extractor = MuseSparkFeatureExtractor(
                sampling_rate=audio.get("sampling_rate", 24000),
                samples_per_token=config.audio_config.samples_per_token,
            )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
            config=config,
        )


install_auto_processor_patch(["muse_spark", "muse_spark_open"], MuseSparkProcessor)
