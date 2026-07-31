"""Processor class for Inkling.

Adapted from HuggingFace Transformers.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
import transformers
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

# dMel quantization defaults (match processor_config.json).
NUM_DMEL_BINS = 16
DMEL_MIN_VALUE = -7.0
DMEL_MAX_VALUE = 2.0


def dmel_bin_centers(
    num_bins: int = NUM_DMEL_BINS,
    min_value: float = DMEL_MIN_VALUE,
    max_value: float = DMEL_MAX_VALUE,
) -> np.ndarray:
    return np.linspace(min_value, max_value, num_bins, dtype=np.float64)


def extract_dmel_bins(
    features,
    bin_centers: np.ndarray,
    min_value: float = DMEL_MIN_VALUE,
    max_value: float = DMEL_MAX_VALUE,
) -> np.ndarray:
    """Quantize log-mel features to nearest-bin dMel token ids.

    Same arithmetic as the reference processor (clamp then nearest bin center,
    computed in float64), but in numpy so audio input does not require torch.
    numpy and torch linspace centers can differ in their last ulp, which could
    only matter for a feature value exactly midway between two centers; real
    log-mel features never sit on that measure-zero set.
    """
    mel = np.clip(np.asarray(features, dtype=np.float64), min_value, max_value)
    return np.abs(mel[..., None] - bin_centers).argmin(axis=-1).astype(np.int32)


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
    """Wraps the Inkling image processor, audio feature extractor and tokenizer.

    ``__call__`` expands the single image (audio) placeholder emitted by the
    chat template into one ``image_token`` per patch (``audio_token`` per dMel
    frame), and tokenizes via ``encode`` (+ manual left/right padding) so it
    does not depend on the tokenizer having a pad token configured. Audio is
    quantized to dMel token ids in numpy; torch is not required.
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
        audio_extractor=None,
        num_dmel_bins: int = NUM_DMEL_BINS,
        dmel_min_value: float = DMEL_MIN_VALUE,
        dmel_max_value: float = DMEL_MAX_VALUE,
        **kwargs,
    ):
        self.image_token = IMAGE_TOKEN
        self.audio_token = AUDIO_TOKEN
        self.num_dmel_bins = num_dmel_bins
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.bin_centers = dmel_bin_centers(
            num_dmel_bins, dmel_min_value, dmel_max_value
        )
        if image_processor is None:
            image_processor = InklingImageProcessor()
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        # Stored under the conventional name (callers probe
        # `processor.feature_extractor` for the sampling rate), but the
        # parameter must not be called that: ProcessorMixin derives its
        # required arguments from modality keywords in the __init__ signature,
        # and would demand a feature extractor for every checkpoint.
        self.feature_extractor = audio_extractor

    def _extract_dmel_bins(self, input_features) -> np.ndarray:
        return extract_dmel_bins(
            input_features, self.bin_centers, self.dmel_min_value, self.dmel_max_value
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput]] = None,
        audio=None,
        padding_side: str = "left",
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")
        kwargs.pop("return_tensors", None)

        image_inputs = {}
        num_patches = None
        if images is not None:
            image_inputs = self.image_processor(images)
            num_patches = image_inputs.pop("num_patches")

        audio_inputs = {}
        num_audio_tokens = None
        if audio is not None:
            if self.feature_extractor is None:
                raise ValueError(
                    "This processor has no audio feature extractor (the "
                    "checkpoint ships none, or the installed transformers "
                    "predates Inkling); audio input is unavailable."
                )
            if not isinstance(audio, list):
                audio = [audio]
            fe_out = self.feature_extractor(
                audio,
                sampling_rate=getattr(self.feature_extractor, "sampling_rate", None),
                return_tensors="np",
            )
            bins = self._extract_dmel_bins(fe_out["input_features"])
            mask = fe_out.get("input_features_mask", fe_out.get("attention_mask"))
            audio_inputs = {"audio_input_ids": mx.array(bins)}
            if mask is not None:
                mask = np.asarray(mask)
                audio_inputs["audio_input_ids_mask"] = mx.array(mask)
            # One audio soft token per valid dMel frame.
            num_audio_tokens = [
                int(mask[i].sum()) if mask is not None else int(bins[i].shape[-2])
                for i in range(bins.shape[0])
            ]

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

        # Audio placeholders expand the same way, one token per dMel frame.
        if num_audio_tokens is not None and text is not None:
            idx = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    text[i] = text[i].replace(
                        self.audio_token,
                        "<|placeholder|>" * num_audio_tokens[idx],
                        1,
                    )
                    idx += 1
                text[i] = text[i].replace("<|placeholder|>", self.audio_token)

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
        names = (
            self.tokenizer.model_input_names + self.image_processor.model_input_names
        )
        return list(dict.fromkeys(names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
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

        # Audio: the checkpoint nests the feature-extractor config inside
        # processor_config.json. Resolve the class from transformers by name;
        # when it is missing (transformers predating Inkling) the processor
        # still loads and raises a clear error only if audio arrives.
        feature_extractor = None
        dmel_kwargs = {}
        proc_cfg_path = model_path / "processor_config.json"
        if proc_cfg_path.exists():
            proc_cfg = json.loads(proc_cfg_path.read_text())
            dmel_kwargs = {
                key: proc_cfg[key]
                for key in ("num_dmel_bins", "dmel_min_value", "dmel_max_value")
                if key in proc_cfg
            }
            fe_cfg = proc_cfg.get("feature_extractor")
            if isinstance(fe_cfg, dict):
                fe_cfg = dict(fe_cfg)
                fe_cls = getattr(
                    transformers, fe_cfg.pop("feature_extractor_type", ""), None
                )
                if fe_cls is not None:
                    feature_extractor = fe_cls(**fe_cfg)

        return cls(
            image_processor=InklingImageProcessor(),
            tokenizer=tokenizer,
            chat_template=chat_template,
            audio_extractor=feature_extractor,
            **dmel_kwargs,
        )


from ..base import install_auto_processor_patch  # noqa: E402

install_auto_processor_patch(["inkling", "inkling_mm_model"], InklingProcessor)
