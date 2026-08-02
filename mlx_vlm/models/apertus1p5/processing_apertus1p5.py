"""Processor for Apertus 1.5.

Ported from the reference implementation. Transformers has no ``apertus1p5``
support yet, so the image processor and the audio feature extraction are
implemented here and the processor is registered on ``AutoProcessor``.

Each ``<|image|>`` placeholder is expanded into the model's structured run

    <|img_start|>{H}*{W}<|img_token_start|>ROW<|img_end_of_row|>...ROW<|img_end|>

where each ``ROW`` is ``grid_width`` ``<|image|>`` tokens, one per discrete
code, and ``{H}*{W}`` is a literal textual size header. Audio expands to

    <|audio_start|>{one <|audio|> per code}<|audio_end|>
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

# Peak level every clip is normalized to before feature extraction.
AUDIO_TARGET_PEAK_DBFS = -3.0


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 256 * 256,
    max_pixels: int = 1400 * 1400,
) -> Tuple[int, int]:
    """Clamp the pixel area to ``[min_pixels, max_pixels]`` at a fixed aspect
    ratio, then round both sides half-up to multiples of ``factor``.

    This reproduces the reference pipeline, including its ``int()`` truncations.
    Each side is floored at ``factor`` so extreme aspect ratios cannot round a
    side down to zero.
    """
    target_area = max(min(max_pixels, height * width), min_pixels)
    aspect_ratio = width / height
    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + factor // 2) // factor) * factor
    new_width = ((new_width + factor // 2) // factor) * factor
    return max(new_height, factor), max(new_width, factor)


class Apertus1p5ImageProcessor:
    """Resizes to a ``spatial_factor`` grid and normalizes to ``[-1, 1]``."""

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        min_pixels: int = 256 * 256,
        max_pixels: int = 1400 * 1400,
        spatial_factor: int = 16,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.spatial_factor = spatial_factor
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]

    def get_number_of_image_patches(self, height: int, width: int) -> int:
        """Number of discrete codes, i.e. placeholder tokens, for an image."""
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.spatial_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return (resized_height // self.spatial_factor) * (
            resized_width // self.spatial_factor
        )

    def preprocess(self, images: List[Image.Image]) -> dict:
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)

        processed, sizes = [], []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image).astype(np.uint8))
            image = image.convert("RGB")
            height, width = smart_resize(
                image.height,
                image.width,
                factor=self.spatial_factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image = image.resize((width, height), Image.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            processed.append((array - mean) / std)
            sizes.append([height, width])

        # Pad the batch to a common size; the model crops it away per image.
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)
        pixel_values = np.zeros(
            (len(processed), max_height, max_width, 3), dtype=np.float32
        )
        for index, array in enumerate(processed):
            pixel_values[index, : array.shape[0], : array.shape[1]] = array

        image_sizes = np.array(sizes, dtype=np.int32)
        image_grids = image_sizes // self.spatial_factor
        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "image_grids": image_grids,
        }


class Apertus1p5AudioFeatureExtractor:
    """Pads mono 24 kHz waveforms and reports the resulting code counts."""

    model_input_names = ["input_features", "feature_attention_mask"]

    def __init__(
        self,
        sampling_rate: int = 24000,
        hop_length: int = 600,
        padding_value: float = 0.0,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.padding_value = padding_value

    def get_num_audio_codes(self, num_samples: int) -> int:
        """``ceil(num_samples / hop_length)``; the model pads internally."""
        return -(-num_samples // self.hop_length)

    def __call__(self, audio: List[np.ndarray]) -> dict:
        target_peak = 10.0 ** (AUDIO_TARGET_PEAK_DBFS / 20.0)
        clips = []
        for clip in audio:
            clip = np.asarray(clip, dtype=np.float32)
            if clip.ndim > 1:
                clip = np.squeeze(clip)
            if clip.ndim != 1:
                raise ValueError(
                    "Apertus 1.5 expects mono audio; got a clip with shape "
                    f"{np.asarray(clip).shape}."
                )
            if clip.size == 0:
                raise ValueError("Apertus 1.5 received an empty audio clip.")
            # Peak-normalize, as the reference pipeline does; the codec feature
            # extractor itself deliberately performs no normalization.
            clips.append(clip * (target_peak / max(float(np.abs(clip).max()), 1e-10)))

        max_length = max(clip.size for clip in clips)
        input_features = np.full(
            (len(clips), max_length, 1), self.padding_value, dtype=np.float32
        )
        feature_attention_mask = np.zeros((len(clips), max_length), dtype=np.int32)
        for index, clip in enumerate(clips):
            input_features[index, : clip.size, 0] = clip
            feature_attention_mask[index, : clip.size] = 1

        return {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "num_audio_codes": [self.get_num_audio_codes(clip.size) for clip in clips],
        }


class Apertus1p5Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        feature_extractor=None,
        chat_template=None,
        **kwargs,
    ):
        self.feature_extractor = feature_extractor or Apertus1p5AudioFeatureExtractor()

        def special(name, default):
            return getattr(tokenizer, name, None) or default

        self.image_token = special("image_token", "<|image|>")
        self.audio_token = special("audio_token", "<|audio|>")
        self.boi_token = special("boi_token", "<|img_start|>")
        self.eoi_token = special("eoi_token", "<|img_end|>")
        self.image_wrapper_token = special("image_wrapper_token", "<|img_token_start|>")
        self.eol_token = special("eol_token", "<|img_end_of_row|>")
        self.boa_token = special("boa_token", "<|audio_start|>")
        self.eoa_token = special("eoa_token", "<|audio_end|>")

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token, token_id in (
            (self.image_token, self.image_token_id),
            (self.audio_token, self.audio_token_id),
        ):
            if token_id is None or (
                unk_token_id is not None and token_id == unk_token_id
            ):
                raise ValueError(
                    f"The tokenizer does not contain the media placeholder token "
                    f"'{token}'. Apertus 1.5 requires a tokenizer carrying the "
                    "media special tokens."
                )

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        hub_kwargs = {
            key: kwargs[key]
            for key in (
                "revision",
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
            )
            if key in kwargs
        }

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        tokenizer = load_chat_template(tokenizer, pretrained_model_name_or_path)

        config = {}
        model_path = Path(pretrained_model_name_or_path)
        config_path = model_path / "processor_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            try:
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    str(pretrained_model_name_or_path),
                    "processor_config.json",
                    **hub_kwargs,
                )
                config = json.loads(Path(downloaded).read_text())
            except Exception:
                config = {}

        return cls(
            image_processor=Apertus1p5ImageProcessor(
                **config.get("image_processor", {})
            ),
            feature_extractor=Apertus1p5AudioFeatureExtractor(
                **config.get("feature_extractor", {})
            ),
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    def replace_image_token(self, image_grids, image_idx: int) -> str:
        grid_height, grid_width = (int(side) for side in image_grids[image_idx])
        rows = self.eol_token.join([self.image_token * grid_width] * grid_height)
        return (
            f"{self.boi_token}{grid_height}*{grid_width}"
            f"{self.image_wrapper_token}{rows}{self.eoi_token}"
        )

    def replace_audio_token(self, num_audio_codes, audio_idx: int) -> str:
        num_codes = int(num_audio_codes[audio_idx])
        return f"{self.boa_token}{self.audio_token * num_codes}{self.eoa_token}"

    def _expand(
        self, text: List[str], token: str, replacements: List[str]
    ) -> List[str]:
        """Consume replacements in batch-sample then left-to-right order."""
        total = sum(sample.count(token) for sample in text)
        if total != len(replacements):
            raise ValueError(
                f"The text contains {total} '{token}' placeholders but "
                f"{len(replacements)} inputs were passed."
            )
        index = 0
        expanded = []
        for sample in text:
            while token in sample:
                sample = sample.replace(token, replacements[index], 1)
                index += 1
            expanded.append(sample)
        return expanded

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        padding=True,
        padding_side="left",
        return_tensors="mlx",
        add_special_tokens=False,
        **kwargs,
    ):
        if text is None:
            raise ValueError(
                "`text` with media placeholder tokens must be provided when "
                "passing images or audio."
            )
        if isinstance(text, str):
            text = [text]
        text = list(text)

        data = {}
        if images is not None and len(images) > 0:
            if not isinstance(images, list):
                images = [images]
            image_inputs = self.image_processor.preprocess(images)
            image_grids = image_inputs.pop("image_grids")
            # The expanded rows must be built before tokenizing, but the
            # placeholder run has to survive as `<|image|>` tokens afterwards.
            text = self._expand(
                text,
                self.image_token,
                [
                    self.replace_image_token(image_grids, index)
                    for index in range(len(images))
                ],
            )
            data.update(image_inputs)

        if audio is not None and len(audio) > 0:
            if not isinstance(audio, list):
                audio = [audio]
            audio_inputs = self.feature_extractor(audio)
            num_audio_codes = audio_inputs.pop("num_audio_codes")
            text = self._expand(
                text,
                self.audio_token,
                [
                    self.replace_audio_token(num_audio_codes, index)
                    for index in range(len(audio))
                ],
            )
            data.update(audio_inputs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            padding_side=padding_side,
            return_tensors="np",
        )
        data["input_ids"] = np.asarray(encoding["input_ids"])
        data["attention_mask"] = np.asarray(encoding["attention_mask"])

        return to_mlx(data) if return_tensors == "mlx" else data

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                list(self.tokenizer.model_input_names)
                + list(self.image_processor.model_input_names)
                + list(self.feature_extractor.model_input_names)
            )
        )


__all__ = ["Apertus1p5Processor", "Apertus1p5ImageProcessor", "smart_resize"]

install_auto_processor_patch("apertus1p5", Apertus1p5Processor)
