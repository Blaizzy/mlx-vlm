"""Processor class for Inkling.

Adapted from HuggingFace Transformers.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

# OpenAI CLIP normalization constants (match processor_config.json).
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_TOKEN = "<|unused_200054|>"
IMAGE_BOS_TOKEN = "<|content_image|>"
AUDIO_TOKEN = "<|unused_200053|>"
AUDIO_BOS_TOKEN = "<|content_audio_input|>"

NUM_DMEL_BINS = 16
DMEL_MIN_VALUE = -7.0
DMEL_MAX_VALUE = 2.0


def dmel_bin_centers(
    num_bins: int = NUM_DMEL_BINS,
    min_value: float = DMEL_MIN_VALUE,
    max_value: float = DMEL_MAX_VALUE,
) -> np.ndarray:
    return np.linspace(min_value, max_value, num_bins, dtype=np.float64)


def dmel_bin_boundaries(
    num_bins: int = NUM_DMEL_BINS,
    min_value: float = DMEL_MIN_VALUE,
    max_value: float = DMEL_MAX_VALUE,
) -> mx.array:
    centers = dmel_bin_centers(num_bins, min_value, max_value)
    midpoints = (centers[:-1] + centers[1:]) / 2
    boundaries = midpoints.astype(np.float32)

    # Metal has no float64. Use the largest float32 value at or below each
    # midpoint so a strict comparison preserves the reference's lower-bin ties.
    rounded_up = boundaries.astype(np.float64) > midpoints
    boundaries[rounded_up] = np.nextafter(boundaries[rounded_up], np.float32(-np.inf))
    return mx.array(boundaries)


def extract_dmel_bins(
    input_features,
    bin_boundaries: Optional[mx.array] = None,
    min_value: float = DMEL_MIN_VALUE,
    max_value: float = DMEL_MAX_VALUE,
    max_frames_per_chunk: int = 1024,
) -> mx.array:
    if max_frames_per_chunk <= 0:
        raise ValueError("max_frames_per_chunk must be positive")

    boundaries = (
        dmel_bin_boundaries(min_value=min_value, max_value=max_value)
        if bin_boundaries is None
        else bin_boundaries
    )
    mel = (
        input_features
        if isinstance(input_features, mx.array)
        else mx.array(input_features)
    )
    mel = mx.clip(mel.astype(mx.float32), min_value, max_value)

    def quantize(chunk):
        bins = mx.zeros(chunk.shape, dtype=mx.int32)
        for boundary in boundaries:
            bins = bins + (chunk > boundary)
        return bins

    if mel.ndim < 2 or mel.shape[-2] <= max_frames_per_chunk:
        return quantize(mel)

    chunks = []
    for start in range(0, mel.shape[-2], max_frames_per_chunk):
        bins = quantize(mel[..., start : start + max_frames_per_chunk, :])
        mx.eval(bins)
        chunks.append(bins)
    return mx.concatenate(chunks, axis=-2)


def divide_to_patches(image: np.ndarray, patch_size: int) -> List[np.ndarray]:
    """Split a ``[C, H, W]`` image into a grid of ``[C, <=patch, <=patch]`` tiles.

    Mirrors the reference ``divide_to_patches``: ``ceil(H/patch)`` rows and
    ``W // patch + 1`` columns (the trailing column is intentional). Edge tiles
    may be smaller than ``patch_size`` and are padded by the caller.
    """
    _, height, width = image.shape
    num_rows = (height + patch_size - 1) // patch_size
    num_cols = width // patch_size + 1
    patches = []
    for i in range(num_rows):
        for j in range(num_cols):
            y, x = i * patch_size, j * patch_size
            patches.append(image[:, y : y + patch_size, x : x + patch_size])
    return patches


class InklingImageProcessor(BaseImageProcessor):
    """Patchifies an image into ``[num_patches, 2, 40, 40, 3]`` soft-token frames."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 40,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        rescale_factor: float = 1 / 255,
        temporal_patch_size: int = 2,
        rescale_image_frac: Optional[float] = None,
        rescale_image_max_upscaled_long_edge: Optional[int] = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.rescale_factor = rescale_factor
        self.temporal_patch_size = temporal_patch_size
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge

    def _preprocess_one(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)  # [C, H, W], 0-255

        if self.rescale_image_frac is not None:
            _, height, width = arr.shape
            long_edge = max(height, width)
            target = long_edge * self.rescale_image_frac
            if self.rescale_image_max_upscaled_long_edge is not None:
                target = min(
                    target, max(self.rescale_image_max_upscaled_long_edge, long_edge)
                )
            ratio = target / long_edge
            if ratio != 1.0:
                new_h = math.floor(height * ratio + 0.5)
                new_w = math.floor(width * ratio + 0.5)
                image = image.resize((new_w, new_h), Image.LANCZOS)
                arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

        patches = divide_to_patches(arr, self.patch_size)
        # Pad each tile to patch_size x patch_size with -1.0 (pre-rescale space).
        padded = np.full(
            (len(patches), 3, self.patch_size, self.patch_size), -1.0, dtype=np.float32
        )
        for k, p in enumerate(patches):
            padded[k, :, : p.shape[1], : p.shape[2]] = p

        # Rescale then per-channel normalize.
        padded *= self.rescale_factor
        mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
        std = np.array(self.image_std, dtype=np.float32)[:, None, None]
        padded = (padded - mean) / std

        # Add a temporal axis of size `temporal_patch_size` and move it to dim 1:
        # [N, C, H, W] -> [N, C, H, W, T] -> [N, T, H, W, C].
        padded = np.repeat(padded[..., None], self.temporal_patch_size, axis=-1)
        return padded.transpose(0, 4, 2, 3, 1)

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        images = make_list_of_images(images)
        per_image = [self._preprocess_one(img) for img in images]
        num_patches = [p.shape[0] for p in per_image]
        pixel_values = np.concatenate(per_image, axis=0)
        return BatchFeature(
            data={
                "pixel_values": mx.array(pixel_values),
                "num_patches": num_patches,
            }
        )

    def __call__(self, images: ImageInput, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class InklingProcessor(ProcessorMixin):
    """Wraps Inkling audio/image preprocessing and tokenization.

    ``__call__`` expands media placeholders to match their image patches or
    audio frames. Text is tokenized via ``encode`` with manual padding so no
    tokenizer pad token is required.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "InklingImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token: str = IMAGE_TOKEN,
        audio_token: str = AUDIO_TOKEN,
        image_bos_token: str = IMAGE_BOS_TOKEN,
        audio_bos_token: str = AUDIO_BOS_TOKEN,
        num_dmel_bins: int = NUM_DMEL_BINS,
        dmel_min_value: float = DMEL_MIN_VALUE,
        dmel_max_value: float = DMEL_MAX_VALUE,
        **kwargs,
    ):
        from .audio_feature_extractor import InklingAudioFeatureExtractor

        feature_extractor = kwargs.pop("feature_extractor", None)
        self.image_token = image_token
        self.audio_token = audio_token
        self.image_bos_token = image_bos_token
        self.audio_bos_token = audio_bos_token
        self.num_dmel_bins = num_dmel_bins
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.bin_boundaries = dmel_bin_boundaries(
            num_dmel_bins, dmel_min_value, dmel_max_value
        )
        if image_processor is None:
            image_processor = InklingImageProcessor()
        if feature_extractor is None:
            feature_extractor = InklingAudioFeatureExtractor()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.feature_extractor = feature_extractor

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput]] = None,
        audio=None,
        padding_side: str = "left",
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None and audio is None:
            raise ValueError(
                "You have to specify at least one of `images`, `text`, or `audio`."
            )
        kwargs.pop("return_tensors", None)

        image_inputs = {}
        num_patches = None
        if images is not None:
            image_inputs = self.image_processor(images)
            num_patches = image_inputs.pop("num_patches")

        audio_inputs = {}
        num_audio_tokens = None
        if audio is not None:
            audio_kwargs = {
                key: kwargs[key]
                for key in (
                    "sampling_rate",
                    "padding",
                    "max_length",
                    "truncation",
                    "pad_to_multiple_of",
                )
                if key in kwargs
            }
            extracted = self.feature_extractor(audio, **audio_kwargs)
            audio_mask = extracted.get("input_features_mask")
            audio_inputs = {
                "audio_input_ids": extract_dmel_bins(
                    extracted["input_features"],
                    self.bin_boundaries,
                    self.dmel_min_value,
                    self.dmel_max_value,
                ),
                "audio_input_ids_mask": audio_mask,
            }
            if audio_mask is not None:
                num_audio_tokens = audio_mask.sum(axis=1).tolist()
            else:
                num_audio_tokens = [
                    audio_inputs["audio_input_ids"].shape[-2]
                ] * audio_inputs["audio_input_ids"].shape[0]

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError("`text` must be a string or a list of strings")

        # Expand each image placeholder into `num_patches` image tokens. The
        # two-step placeholder swap avoids re-matching freshly inserted tokens.
        if num_patches is not None and text is not None:
            idx = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * int(num_patches[idx]),
                        1,
                    )
                    idx += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if num_audio_tokens is not None and text is not None:
            idx = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    text[i] = text[i].replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * int(num_audio_tokens[idx]),
                        1,
                    )
                    idx += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        text_inputs = {}
        if text is not None:
            all_ids = [self.tokenizer.encode(t) for t in text]
            max_len = max(len(ids) for ids in all_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids, attention = [], []
            for ids in all_ids:
                pad = [pad_id] * (max_len - len(ids))
                if padding_side == "left":
                    input_ids.append(pad + ids)
                    attention.append([0] * len(pad) + [1] * len(ids))
                else:
                    input_ids.append(ids + pad)
                    attention.append([1] * len(ids) + [0] * len(pad))
            text_inputs = {
                "input_ids": mx.array(input_ids),
                "attention_mask": mx.array(attention),
            }

        return BatchFeature(data={**text_inputs, **image_inputs, **audio_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = [
            "audio_input_ids",
            "audio_input_ids_mask",
            *self.tokenizer.model_input_names,
            *self.image_processor.model_input_names,
        ]
        return list(dict.fromkeys(names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from .audio_feature_extractor import InklingAudioFeatureExtractor

        kwargs.pop("trust_remote_code", None)
        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # The Inkling tokenizer declares no pad/eos string
        # TODO: fix in the tokenizer repo on the Hugging Face Hub
        if getattr(tokenizer, "pad_token", None) is None:
            eos_id = None
            cfg_path = model_path / "config.json"
            if cfg_path.exists():
                eos_id = json.loads(cfg_path.read_text()).get("eos_token_id")
            if isinstance(eos_id, (list, tuple)):
                eos_id = eos_id[0]
            if eos_id is not None:
                eos_str = tokenizer.convert_ids_to_tokens(eos_id)
                if getattr(tokenizer, "eos_token", None) is None:
                    tokenizer.eos_token = eos_str
                tokenizer.pad_token = eos_str

        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            jinja_path = model_path / "chat_template.jinja"
            if jinja_path.exists():
                chat_template = jinja_path.read_text(encoding="utf-8")
                tokenizer.chat_template = chat_template

        processor_config = {}
        processor_config_path = model_path / "processor_config.json"
        if processor_config_path.exists():
            processor_config = json.loads(processor_config_path.read_text())
        elif not is_local:
            try:
                from huggingface_hub import hf_hub_download

                processor_config_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "processor_config.json"
                    )
                )
                processor_config = json.loads(processor_config_path.read_text())
            except Exception:
                pass

        feature_config = dict(processor_config.get("feature_extractor", {}))
        feature_config.pop("feature_extractor_type", None)

        return cls(
            image_processor=InklingImageProcessor(),
            tokenizer=tokenizer,
            chat_template=chat_template,
            feature_extractor=InklingAudioFeatureExtractor(**feature_config),
            image_token=processor_config.get("image_token", IMAGE_TOKEN),
            audio_token=processor_config.get("audio_token", AUDIO_TOKEN),
            image_bos_token=processor_config.get("image_bos_token", IMAGE_BOS_TOKEN),
            audio_bos_token=processor_config.get("audio_bos_token", AUDIO_BOS_TOKEN),
            num_dmel_bins=processor_config.get("num_dmel_bins", NUM_DMEL_BINS),
            dmel_min_value=processor_config.get("dmel_min_value", DMEL_MIN_VALUE),
            dmel_max_value=processor_config.get("dmel_max_value", DMEL_MAX_VALUE),
        )


from ..base import install_auto_processor_patch  # noqa: E402

install_auto_processor_patch(["inkling", "inkling_mm_model"], InklingProcessor)
