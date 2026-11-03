"""NumPy/Pillow preprocessing for MiniCPM-V 4.7, without PyTorch."""

import json
from pathlib import Path

import numpy as np
from transformers.utils.hub import cached_file

from ..base import install_auto_processor_patch, load_chat_template
from ..minicpmv4_6.processing_minicpmv4_6 import (
    MiniCPMVImageProcessor,
    MiniCPMVProcessor,
    MiniCPMVVideoProcessor,
    _extract_hf_from_pretrained_kwargs,
    _load_local_qwen2_tokenizer,
)


class MiniCPMV4_7Processor(MiniCPMVProcessor):
    model_type = "minicpmv4_7"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    @staticmethod
    def _strip_think_prefix(text):
        # Keep the checkpoint's complete thinking/non-thinking prompt.
        return text

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def _build_placeholder_ids_for_video(
        self,
        video_target_sizes,
        frame_grids,
        num_patches_per_frame,
        video_idx,
        token_divisor,
    ):
        ids = super()._build_placeholder_ids_for_video(
            video_target_sizes,
            frame_grids,
            num_patches_per_frame,
            video_idx,
            token_divisor,
        )
        # 4.7 assigns image IDs only to still images, never video clips.
        return ids[len(self._build_image_id_ids(video_idx)) :]

    def __call__(self, text=None, images=None, videos=None, **kwargs):
        mode = kwargs.pop("downsample_mode", self.image_processor.downsample_mode)
        if mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
        texts = self._normalize_text(text)
        for inputs, normalize, count, name in (
            (images, self._normalize_images, self._count_image_markers, "image"),
            (videos, self._normalize_videos, self._count_video_markers, "video"),
        ):
            samples = normalize(inputs, len(texts), texts)
            if any(
                count(prompt) != len(sample) for prompt, sample in zip(texts, samples)
            ):
                raise ValueError(
                    f"Number of {name} placeholders must match {name} inputs."
                )
        old_modes = (
            self.image_processor.downsample_mode,
            self.video_processor.downsample_mode,
        )
        self.image_processor.downsample_mode = self.video_processor.downsample_mode = (
            mode
        )
        try:
            result = super().__call__(
                text=texts, images=images, videos=videos, **kwargs
            )
        finally:
            (
                self.image_processor.downsample_mode,
                self.video_processor.downsample_mode,
            ) = old_modes
        ids = result["input_ids"]
        result["mm_token_type_ids"] = np.where(
            ids == self.image_token_id, 1, np.where(ids == self.video_token_id, 2, 0)
        ).astype(np.int32)
        result["downsample_mode"] = mode
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        hf_kwargs = _extract_hf_from_pretrained_kwargs(kwargs)

        def read_config(filename):
            path = cached_file(
                pretrained_model_name_or_path,
                filename,
                _raise_exceptions_for_missing_entries=False,
                **hf_kwargs,
            )
            return json.loads(Path(path).read_text()) if path else {}

        config = read_config("config.json")
        image_config = read_config("preprocessor_config.json")
        processor_config = read_config("processor_config.json")
        image_config.update(processor_config.get("image_processor", {}))
        image_config.setdefault("downsample_mode", config.get("downsample_mode", "16x"))
        video_config = {**image_config, **processor_config.get("video_processor", {})}
        for options in (image_config, video_config):
            options["norm_mean"] = options.pop(
                "image_mean", options.get("norm_mean", [0.5] * 3)
            )
            options["norm_std"] = options.pop(
                "image_std", options.get("norm_std", [0.5] * 3)
            )
        tokenizer = _load_local_qwen2_tokenizer(
            pretrained_model_name_or_path, hf_kwargs=hf_kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        return cls(
            image_processor=MiniCPMVImageProcessor(**image_config),
            video_processor=MiniCPMVVideoProcessor(**video_config),
            tokenizer=tokenizer,
            chat_template=kwargs.get("chat_template", tokenizer.chat_template),
        )

    def save_pretrained(self, save_directory, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path, **kwargs)

        def component_config(component):
            fields = (
                "max_slice_nums",
                "patch_size",
                "scale_resolution",
                "slice_mode",
                "use_image_id",
                "downsample_mode",
                "pack_image_by_patch",
                "max_num_frames",
                "stack_frames",
            )
            return {
                **{
                    key: getattr(component, key)
                    for key in fields
                    if hasattr(component, key)
                },
                "image_mean": component.mean.reshape(-1).tolist(),
                "image_std": component.std.reshape(-1).tolist(),
            }

        config = {
            "processor_class": type(self).__name__,
            "image_processor": component_config(self.image_processor),
            "video_processor": component_config(self.video_processor),
        }
        (path / "processor_config.json").write_text(json.dumps(config, indent=2) + "\n")
        return [str(path / "processor_config.json")]


install_auto_processor_patch("minicpmv4_7", MiniCPMV4_7Processor)
