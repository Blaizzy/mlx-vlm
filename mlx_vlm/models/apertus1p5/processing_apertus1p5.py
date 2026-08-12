import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

_HUB_DOWNLOAD_KWARGS = {
    "cache_dir",
    "endpoint",
    "etag_timeout",
    "force_download",
    "headers",
    "library_name",
    "library_version",
    "local_files_only",
    "repo_type",
    "revision",
    "subfolder",
    "token",
    "user_agent",
}


# Resizing must be bit-exact with Torchvision's uint8 CPU bicubic path, not
# merely close: the vision tokenizer maps pixels to discrete codes via a hard
# argmax over a 131k-entry codebook, so sub-pixel differences from PIL's
# bicubic filter can flip codes and change the tokens the model sees.
def _cubic_filter(value: float) -> float:
    """Keys' cubic filter used by Torchvision for antialiased bicubic resize."""
    value = abs(value)
    if value < 1.0:
        return ((1.5 * value - 2.5) * value * value) + 1.0
    if value < 2.0:
        return (((-0.5 * value + 2.5) * value - 4.0) * value) + 2.0
    return 0.0


def _uint8_bicubic_coefficients(
    input_size: int, output_size: int
) -> tuple[list[int], list[np.ndarray], int]:
    """Build the integer coefficients used by Torchvision's uint8 CPU path."""
    scale = input_size / output_size
    support = 2.0 * scale if scale >= 1.0 else 2.0
    max_interp_size = math.ceil(support) * 2 + 1
    starts = []
    weights = []
    max_weight = 0.0

    for output_index in range(output_size):
        center = scale * (output_index + 0.5)
        inverse_scale = 1.0 / scale if scale >= 1.0 else 1.0
        start = max(int(center - support + 0.5), 0)
        count = min(int(center + support + 0.5), input_size) - start
        count = max(0, min(count, max_interp_size))
        coefficients = [
            _cubic_filter((index + start - center + 0.5) * inverse_scale)
            for index in range(count)
        ]
        total = sum(coefficients)
        if total:
            coefficients = [coefficient / total for coefficient in coefficients]
        max_weight = max(max_weight, max(coefficients, default=0.0))
        starts.append(start)
        weights.append(np.asarray(coefficients, dtype=np.float64))

    precision = 0
    for precision in range(22):
        next_value = int(0.5 + max_weight * (1 << (precision + 1)))
        if next_value >= 1 << 15:
            break

    integer_weights = []
    multiplier = 1 << precision
    for coefficients in weights:
        integer_weights.append(
            np.asarray(
                [
                    (
                        int(-0.5 + coefficient * multiplier)
                        if coefficient < 0
                        else int(0.5 + coefficient * multiplier)
                    )
                    for coefficient in coefficients
                ],
                dtype=np.int64,
            )
        )
    return starts, integer_weights, precision


def _resize_uint8_axis(image: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    starts, weights, precision = _uint8_bicubic_coefficients(
        image.shape[axis], output_size
    )
    output_shape = list(image.shape)
    output_shape[axis] = output_size
    output = np.empty(output_shape, dtype=np.uint8)
    rounding = 1 << (precision - 1)

    for output_index, (start, coefficients) in enumerate(zip(starts, weights)):
        indices = np.arange(start, start + len(coefficients))
        values = np.take(image, indices, axis=axis).astype(np.int64)
        weight_shape = [1] * image.ndim
        weight_shape[axis] = len(coefficients)
        values = (values * coefficients.reshape(weight_shape)).sum(axis=axis)
        values = (values + rounding) >> precision
        output_slice = [slice(None)] * image.ndim
        output_slice[axis] = output_index
        output[tuple(output_slice)] = np.clip(values, 0, 255).astype(np.uint8)
    return output


def _resize_uint8_bicubic(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a CHW uint8 image exactly like Torchvision's bicubic CPU path."""
    if image.shape[-1] != width:
        image = _resize_uint8_axis(image, width, axis=2)
    if image.shape[-2] != height:
        image = _resize_uint8_axis(image, height, axis=1)
    return image


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 256 * 256,
    max_pixels: int = 1400 * 1400,
) -> tuple[int, int]:
    target_area = max(min(max_pixels, height * width), min_pixels)
    aspect_ratio = width / height
    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + factor // 2) // factor) * factor
    new_width = ((new_width + factor // 2) // factor) * factor
    return max(new_height, factor), max(new_width, factor)


def _is_nested_images(images) -> bool:
    return (
        isinstance(images, (list, tuple))
        and bool(images)
        and all(isinstance(sample, (list, tuple)) for sample in images)
    )


def _is_nested_audio(audio) -> bool:
    # One sub-list of clips per text sample. A bare list of numbers is a
    # single audio clip, not a nesting; a list of arrays is a flat clip list.
    if not isinstance(audio, (list, tuple)) or not audio:
        return False

    def _is_clip_container(item):
        return isinstance(item, (list, tuple)) and (
            len(item) == 0 or not np.isscalar(item[0])
        )

    return all(_is_clip_container(item) for item in audio)


class Apertus1p5FeatureExtractor(FeatureExtractionMixin):
    """Numpy port of the WavTokenizer feature extractor.

    Audio is not resampled or downmixed here; clips must already be mono at
    ``sampling_rate``. Batches are right-padded with ``padding_value`` to the
    longest clip and a padding mask marks the valid samples. The codec model
    pads internally, so each clip produces ``ceil(num_samples / hop_length)``
    codes regardless of batch padding (the model re-slices clips by mask
    before encoding).
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        hop_length: int = 600,
        **kwargs,
    ):
        if feature_size != 1:
            raise ValueError(
                "The Apertus 1.5 audio feature extractor supports mono audio "
                f"only (feature_size=1), got feature_size={feature_size}."
            )
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.hop_length = hop_length

    def get_num_audio_codes(self, num_samples: int) -> int:
        return -(-num_samples // self.hop_length)

    def __call__(self, audio, sampling_rate=None) -> dict:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                "Apertus 1.5 audio must be sampled at "
                f"{self.sampling_rate} Hz, got {sampling_rate} Hz."
            )
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            audio = [audio]
        elif not isinstance(audio, (list, tuple)):
            audio = [audio]
        elif audio and np.isscalar(audio[0]):
            # A bare list of numbers is one clip.
            audio = [audio]

        clips = []
        for index, clip in enumerate(audio):
            clip = np.asarray(clip, dtype=np.float32)
            if clip.ndim != 1:
                raise ValueError(
                    "Expected mono audio of shape (num_samples,) but got "
                    f"shape {clip.shape} at index {index}."
                )
            if clip.size == 0:
                raise ValueError(f"Input audio at index {index} is empty.")
            clips.append(clip)

        max_length = max(clip.size for clip in clips)
        input_values = np.full(
            (len(clips), 1, max_length), self.padding_value, dtype=np.float32
        )
        padding_mask = np.zeros((len(clips), max_length), dtype=np.int64)
        for index, clip in enumerate(clips):
            input_values[index, 0, : clip.size] = clip
            padding_mask[index, : clip.size] = 1
        return {"input_values": input_values, "padding_mask": padding_mask}


class Apertus1p5ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        do_pad: bool = True,
        rescale_factor: float = 1 / 255,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        min_pixels: int = 256 * 256,
        max_pixels: int = 1400 * 1400,
        spatial_factor: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.do_pad = do_pad
        self.rescale_factor = rescale_factor
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.spatial_factor = spatial_factor

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        array = np.asarray(image)
        # Transpose CHW to HWC only when unambiguous: a short HWC image such
        # as (4, W, 3) must not be mistaken for channel-first.
        if (
            array.ndim == 3
            and array.shape[0] in (1, 3, 4)
            and array.shape[-1] not in (1, 3, 4)
        ):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                "Apertus 1.5 image arrays must contain unscaled uint8 pixels; "
                "pass a PIL image or a uint8 NumPy array."
            )
        if array.dtype != np.uint8:
            raise TypeError(
                f"Apertus 1.5 image arrays must have dtype uint8, got {array.dtype}."
            )
        return Image.fromarray(array)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        # Accept the standard transformers image-input forms (single image,
        # flat/nested lists, 4-D batched arrays), like the upstream processor.
        images = make_flat_list_of_images(images)
        if not images:
            raise ValueError(
                "`images` contains no images; pass `images=None` when there "
                "are no images."
            )

        do_resize = kwargs.pop("do_resize", self.do_resize)
        do_rescale = kwargs.pop("do_rescale", self.do_rescale)
        do_normalize = kwargs.pop("do_normalize", self.do_normalize)
        do_convert_rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
        do_pad = kwargs.pop("do_pad", self.do_pad)
        min_pixels = kwargs.pop("min_pixels", self.min_pixels)
        max_pixels = kwargs.pop("max_pixels", self.max_pixels)
        spatial_factor = kwargs.pop("spatial_factor", self.spatial_factor)

        processed = []
        image_sizes = []
        image_grids = []
        for image in images:
            image = self._to_pil(image)
            if do_convert_rgb:
                image = image.convert("RGB")
            array = np.ascontiguousarray(
                np.moveaxis(np.asarray(image, dtype=np.uint8), -1, 0)
            )
            if do_resize:
                height, width = image.height, image.width
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=spatial_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                array = _resize_uint8_bicubic(array, resized_height, resized_width)

            array = array.astype(np.float32)
            if do_rescale:
                array *= self.rescale_factor
            if do_normalize:
                mean = np.asarray(self.image_mean, dtype=np.float32)
                std = np.asarray(self.image_std, dtype=np.float32)
                array = (array - mean[:, None, None]) / std[:, None, None]
            processed.append(array)
            height, width = array.shape[-2:]
            image_sizes.append((height, width))
            image_grids.append((height // spatial_factor, width // spatial_factor))

        if do_pad:
            max_height = max(image.shape[-2] for image in processed)
            max_width = max(image.shape[-1] for image in processed)
            padded = np.zeros(
                (len(processed), 3, max_height, max_width), dtype=np.float32
            )
            for index, image in enumerate(processed):
                padded[index, :, : image.shape[-2], : image.shape[-1]] = image
            pixel_values = padded
        else:
            first_shape = processed[0].shape
            if any(image.shape != first_shape for image in processed):
                raise ValueError(
                    "do_pad=False requires images with identical processed "
                    "sizes; the model consumes a single stacked pixel array."
                )
            pixel_values = np.stack(processed).astype(np.float32)

        data = {
            "pixel_values": pixel_values,
            "image_sizes": np.asarray(image_sizes, dtype=np.int64),
            "image_grids": np.asarray(image_grids, dtype=np.int64),
        }
        return BatchFeature(data=to_mlx(data))

    __call__ = preprocess

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ) -> int:
        images_kwargs = images_kwargs or {}
        factor = images_kwargs.get("spatial_factor", self.spatial_factor)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=images_kwargs.get("min_pixels", self.min_pixels),
            max_pixels=images_kwargs.get("max_pixels", self.max_pixels),
        )
        return (resized_height // factor) * (resized_width // factor)


class Apertus1p5Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "feature_extractor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "AutoFeatureExtractor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        feature_extractor=None,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image|>"
        self.audio_token = getattr(tokenizer, "audio_token", None) or "<|audio|>"
        self.boi_token = getattr(tokenizer, "boi_token", None) or "<|img_start|>"
        self.eoi_token = getattr(tokenizer, "eoi_token", None) or "<|img_end|>"
        self.image_wrapper_token = (
            getattr(tokenizer, "image_wrapper_token", None) or "<|img_token_start|>"
        )
        self.eol_token = getattr(tokenizer, "eol_token", None) or "<|img_end_of_row|>"
        self.boa_token = getattr(tokenizer, "boa_token", None) or "<|audio_start|>"
        self.eoa_token = getattr(tokenizer, "eoa_token", None) or "<|audio_end|>"
        super().__init__(
            image_processor,
            tokenizer,
            feature_extractor or Apertus1p5FeatureExtractor(),
            chat_template=chat_template or getattr(tokenizer, "chat_template", None),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("quantize_activations", None)
        kwargs.pop("eos_token_ids", None)
        kwargs.pop("trust_remote_code", None)
        hub_kwargs = {
            key: value for key, value in kwargs.items() if key in _HUB_DOWNLOAD_KWARGS
        }
        if "token" not in hub_kwargs and "use_auth_token" in kwargs:
            hub_kwargs["token"] = kwargs["use_auth_token"]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False, **kwargs
        )

        processor_config = {}
        model_config = {}
        path = Path(pretrained_model_name_or_path)
        config_root = path
        if path.is_dir() and kwargs.get("subfolder"):
            config_root = path / kwargs["subfolder"]
        load_chat_template(tokenizer, config_root if path.is_dir() else path)

        if config_root.is_dir() and (config_root / "processor_config.json").exists():
            processor_config = json.loads(
                (config_root / "processor_config.json").read_text()
            )
            config_path = config_root / "config.json"
            if config_path.exists():
                model_config = json.loads(config_path.read_text())
        elif not path.exists():
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                pretrained_model_name_or_path,
                "processor_config.json",
                **hub_kwargs,
            )
            processor_config = json.loads(Path(config_path).read_text())
            model_config_path = hf_hub_download(
                pretrained_model_name_or_path,
                "config.json",
                **hub_kwargs,
            )
            model_config = json.loads(Path(model_config_path).read_text())

        image_config = processor_config.get("image_processor", processor_config)
        image_config = {
            key: value
            for key, value in image_config.items()
            if key
            not in {
                "image_processor_type",
                "resample",
                "size",
                "crop_size",
            }
        }
        image_processor = Apertus1p5ImageProcessor(**image_config)
        channel_multiplier = model_config.get("vision_tokenizer_config", {}).get(
            "channel_multiplier"
        )
        if channel_multiplier:
            vision_spatial_factor = 2 ** (len(channel_multiplier) - 1)
            if image_processor.spatial_factor != vision_spatial_factor:
                raise ValueError(
                    "Processor and vision tokenizer spatial factors do not match: "
                    f"{image_processor.spatial_factor} and {vision_spatial_factor}."
                )

        audio_config = {
            key: value
            for key, value in processor_config.get("feature_extractor", {}).items()
            if key
            not in {
                "feature_extractor_type",
                "padding_side",
                "return_attention_mask",
            }
        }
        feature_extractor = Apertus1p5FeatureExtractor(**audio_config)
        upsampling_ratios = model_config.get("audio_tokenizer_config", {}).get(
            "upsampling_ratios"
        )
        if upsampling_ratios:
            codec_hop = 1
            for ratio in upsampling_ratios:
                codec_hop *= ratio
            if feature_extractor.hop_length != codec_hop:
                raise ValueError(
                    "Processor and audio tokenizer hop lengths do not match: "
                    f"{feature_extractor.hop_length} and {codec_hop}."
                )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

    def replace_image_token(self, grid_height: int, grid_width: int) -> str:
        rows = self.eol_token.join([self.image_token * grid_width] * grid_height)
        return (
            f"{self.boi_token}{grid_height}*{grid_width}"
            f"{self.image_wrapper_token}{rows}{self.eoi_token}"
        )

    def replace_audio_token(self, num_codes: int) -> str:
        return f"{self.boa_token}{self.audio_token * num_codes}{self.eoa_token}"

    def _process_audio(self, audio, sampling_rate=None):
        """Peak-normalize clips to -3 dBFS and extract padded features.

        Returns the model inputs (``input_features``/``feature_attention_mask``)
        and the per-clip placeholder expansions. The normalization matches the
        reference Apertus 1.5 pipeline; the feature extractor itself performs
        none. Code counts come from the extractor output masks so they can
        never desync from the features the model will encode.
        """
        target_peak = 10.0 ** (-3.0 / 20.0)
        clips = []
        for clip in audio:
            clip = np.asarray(clip, dtype=np.float32)
            if clip.size > 0:
                clip = clip * (target_peak / max(float(np.abs(clip).max()), 1e-10))
            clips.append(clip)  # empty clips get the extractor's clear error

        extracted = self.feature_extractor(clips, sampling_rate=sampling_rate)
        audio_inputs = {
            "input_features": extracted["input_values"],
            "feature_attention_mask": extracted["padding_mask"],
        }
        replacements = [
            self.replace_audio_token(
                self.feature_extractor.get_num_audio_codes(int(mask.sum()))
            )
            for mask in extracted["padding_mask"]
        ]
        return audio_inputs, replacements

    # These must stay named parameters: utils.process_inputs only forwards
    # generate() kwargs whose names appear in this signature.
    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        add_special_tokens=True,
        sampling_rate=None,
        do_resize=None,
        do_rescale=None,
        do_normalize=None,
        do_convert_rgb=None,
        do_pad=None,
        min_pixels=None,
        max_pixels=None,
        spatial_factor=None,
        **kwargs,
    ):
        if text is None:
            raise ValueError("text must be provided")
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, (list, tuple)) or not all(
            isinstance(sample, str) for sample in text
        ):
            raise TypeError("text must be a string or a sequence of strings")
        if len(text) != 1:
            raise ValueError(
                "The initial Apertus 1.5 implementation supports batch size 1 only."
            )

        nested_images = _is_nested_images(images)
        if nested_images:
            if len(images) != len(text):
                raise ValueError(
                    "Nested image inputs must provide one image list per text sample."
                )
            placeholder_counts = [sample.count(self.image_token) for sample in text]
            image_counts = [len(sample) for sample in images]
            if placeholder_counts != image_counts:
                raise ValueError(
                    "Per-sample image placeholder counts must match the nested "
                    "image inputs."
                )
            images = [image for sample in images for image in sample]
            if not images:
                images = None
        elif images is not None:
            images = make_flat_list_of_images(images)
            if not images:
                images = None

        if audio is not None:
            if isinstance(audio, np.ndarray) and audio.ndim == 1:
                audio = [audio]
            elif not isinstance(audio, (list, tuple)):
                # Non-1D arrays become a single clip; the feature extractor
                # rejects them with a clear mono-audio error.
                audio = [audio]
            elif audio and np.isscalar(audio[0]):
                audio = [np.asarray(audio, dtype=np.float32)]
            if _is_nested_audio(audio):
                if len(audio) != len(text):
                    raise ValueError(
                        "Nested audio inputs must provide one clip list per "
                        "text sample."
                    )
                audio_placeholder_counts = [
                    sample.count(self.audio_token) for sample in text
                ]
                clip_counts = [len(sample) for sample in audio]
                if audio_placeholder_counts != clip_counts:
                    raise ValueError(
                        "Per-sample audio placeholder counts must match the "
                        "nested audio inputs."
                    )
                audio = [clip for sample in audio for clip in sample]
            if not len(audio):
                audio = None

        image_kwargs = {
            key: value
            for key, value in (
                ("do_resize", do_resize),
                ("do_rescale", do_rescale),
                ("do_normalize", do_normalize),
                ("do_convert_rgb", do_convert_rgb),
                ("do_pad", do_pad),
                ("min_pixels", min_pixels),
                ("max_pixels", max_pixels),
                ("spatial_factor", spatial_factor),
            )
            if value is not None
        }

        if images is not None:
            image_inputs = self.image_processor(images, **image_kwargs)
            image_grids = image_inputs.pop("image_grids")
            grids = iter(image_grids.tolist())
            image_count = image_grids.shape[0]
        else:
            image_inputs = {}
            grids = iter(())
            image_count = 0

        if sum(sample.count(self.image_token) for sample in text) != image_count:
            raise ValueError(
                "The number of image placeholders must match the number of images."
            )

        if audio is not None:
            audio_inputs, audio_replacements = self._process_audio(
                audio, sampling_rate=sampling_rate
            )
        else:
            audio_inputs, audio_replacements = {}, []
        if sum(sample.count(self.audio_token) for sample in text) != len(
            audio_replacements
        ):
            raise ValueError(
                "The number of audio placeholders must match the number of "
                "audio clips."
            )

        expanded = []
        for sample in text:
            pieces = sample.split(self.image_token)
            rebuilt = [pieces[0]]
            for piece in pieces[1:]:
                grid_height, grid_width = next(grids)
                rebuilt.extend(
                    (
                        self.replace_image_token(int(grid_height), int(grid_width)),
                        piece,
                    )
                )
            expanded.append("".join(rebuilt))

        # Expand audio after images: image expansions introduce no audio
        # placeholders, so per-clip order is preserved.
        replacements = iter(audio_replacements)
        audio_expanded = []
        for sample in expanded:
            pieces = sample.split(self.audio_token)
            rebuilt = [pieces[0]]
            for piece in pieces[1:]:
                rebuilt.extend((next(replacements), piece))
            audio_expanded.append("".join(rebuilt))
        expanded = audio_expanded

        kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(
            expanded, add_special_tokens=add_special_tokens, **kwargs
        )
        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **audio_inputs})
        )

    @property
    def model_input_names(self):
        # __call__ renames the feature extractor's input_values/padding_mask
        # to the names the model consumes, so advertise those.
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_sizes",
            "input_features",
            "feature_attention_mask",
        ]

    @staticmethod
    def _with_plain_non_user_roles(conversation):
        # The published chat template accepts content-part lists only for
        # user messages; other roles' lists must collapse to plain strings
        # or the template raises jinja2.TemplateError.
        if not isinstance(conversation, list):
            return conversation
        normalized = []
        for message in conversation:
            content = message.get("content") if isinstance(message, dict) else None
            if (
                isinstance(content, list)
                and message.get("role") != "user"
                and message.get("role") != "tool"
            ):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                message = {**message, "content": text}
            normalized.append(message)
        return normalized

    def apply_chat_template(self, conversation, *args, **kwargs):
        return self.tokenizer.apply_chat_template(
            self._with_plain_non_user_roles(conversation), *args, **kwargs
        )

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


install_auto_processor_patch("apertus1p5", Apertus1p5Processor)


__all__ = [
    "Apertus1p5FeatureExtractor",
    "Apertus1p5ImageProcessor",
    "Apertus1p5Processor",
    "smart_resize",
]
