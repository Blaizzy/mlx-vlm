"""Processor for SoMark OXR, matching the official vLLM plugin (oxr_vllm.oxr_processor)."""

import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import cached_file

from ..base import install_auto_processor_patch, load_chat_template

# Defaults used by the official vLLM serving path.
MIN_PIXELS = 200704  # 448 * 448
MAX_PIXELS = 921600  # 960 * 960


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
):
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image if image.mode == "RGB" else image.convert("RGB")


class OXRImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        patch_size: int = 16,
        hidden_stride: int = 2,
        temporal_patch_size: int = 1,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.hidden_stride = hidden_stride
        self.merge_size = hidden_stride
        self.temporal_patch_size = temporal_patch_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def preprocess_one(self, image: Image.Image, min_pixels=None, max_pixels=None):
        max_pixels = max_pixels or self.max_pixels
        min_pixels = min(min_pixels or self.min_pixels, max_pixels)
        image = _to_rgb(image)
        p, s, tp = self.patch_size, self.hidden_stride, self.temporal_patch_size
        width, height = image.size
        rh, rw = smart_resize(height, width, p * s, min_pixels, max_pixels)
        arr = np.asarray(image.resize((rw, rh), resample=Image.BILINEAR), np.float32)
        arr = (arr / 255.0 - self.image_mean) / self.image_std
        arr = arr.transpose(2, 0, 1)[None]  # (1, C, H, W)
        if tp > 1:
            arr = np.repeat(arr, tp, axis=0)
        c = arr.shape[1]
        gt, gh, gw = arr.shape[0] // tp, rh // p, rw // p
        patches = arr.reshape(gt, tp, c, gh // s, s, p, gw // s, s, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = patches.reshape(gt * gh * gw, c * tp * p * p)
        return np.ascontiguousarray(patches), (gt, gh, gw)

    def num_image_tokens(self, grid) -> int:
        return int(np.prod(grid)) // self.hidden_stride**2 // self.temporal_patch_size

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(
        self, images, min_pixels=None, max_pixels=None, return_tensors=None, **kwargs
    ):
        if not isinstance(images, (list, tuple)):
            images = [images]
        patches, grids = [], []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            x, grid = self.preprocess_one(image, min_pixels, max_pixels)
            patches.append(x)
            grids.append(grid)
        return BatchFeature(
            data={
                "pixel_values": np.concatenate(patches, axis=0),
                "image_grid_thw": np.array(grids, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )


class OXRProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        if image_processor is None:
            image_processor = OXRImageProcessor(**kwargs)
        self.image_token = "<image>"  # placeholder rendered by OXR's chat template
        self.image_pad_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images=None,
        text: Union[str, List[str]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", None)
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor.preprocess(
                images, min_pixels=min_pixels, max_pixels=max_pixels
            )
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        text = list(text)
        if images is not None:
            grids = image_inputs["image_grid_thw"]
            index = 0
            for i, t in enumerate(text):
                parts = t.split(self.image_token)
                out = parts[0]
                for part in parts[1:]:
                    if index >= len(grids):
                        raise ValueError("More <image> placeholders than images")
                    n = self.image_processor.num_image_tokens(grids[index])
                    out += (
                        self.vision_start_token
                        + self.image_pad_token * n
                        + self.vision_end_token
                        + part
                    )
                    index += 1
                text[i] = out
            if index != len(grids):
                raise ValueError(
                    f"Got {len(grids)} images but {index} <image> placeholders"
                )

        kwargs.pop("add_special_tokens", None)
        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=return_tensors
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        names = self.tokenizer.model_input_names if self.tokenizer else []
        return list(dict.fromkeys(names + self.image_processor.model_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)
        image_kwargs = {
            key: kwargs.pop(key)
            for key in (
                "min_pixels",
                "max_pixels",
                "patch_size",
                "hidden_stride",
                "temporal_patch_size",
                "image_mean",
                "image_std",
            )
            if key in kwargs
        }
        hub_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "token",
                "subfolder",
            )
            if key in kwargs
        }
        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        # OXR uses the built-in Qwen2 tokenizer; its remote code is torch-only.
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=False,
            **kwargs,
        )
        if is_local:
            load_chat_template(tokenizer, model_path)
        else:
            try:
                template = hf_hub_download(
                    pretrained_model_name_or_path, "chat_template.jinja", **hub_kwargs
                )
                tokenizer.chat_template = Path(template).read_text()
            except Exception:
                pass

        config_path = cached_file(
            pretrained_model_name_or_path,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            **hub_kwargs,
        )
        config = json.loads(Path(config_path).read_text()) if config_path else {}
        vit = config.get("vision_config") or config.get("vit_config", {})

        # Transformers saves nested processor settings in newer versions and
        # standalone image processor settings in older versions.
        image_config = {}
        for filename in ("processor_config.json", "preprocessor_config.json"):
            saved_path = cached_file(
                pretrained_model_name_or_path,
                filename,
                _raise_exceptions_for_missing_entries=False,
                **hub_kwargs,
            )
            if saved_path is None:
                continue
            saved_config = json.loads(Path(saved_path).read_text())
            if filename == "processor_config.json":
                saved_config = saved_config.get("image_processor")
            if saved_config is not None:
                image_config = saved_config
                break
        for key, default in (
            ("patch_size", 16),
            ("hidden_stride", 2),
            ("temporal_patch_size", 1),
        ):
            image_config.setdefault(key, vit.get(key, default))
        image_config.update(image_kwargs)
        image_processor = OXRImageProcessor(**image_config)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


install_auto_processor_patch("oxr", OXRProcessor)

__all__ = ["OXRImageProcessor", "OXRProcessor"]
