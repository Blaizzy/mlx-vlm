import math
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def _convert_to_rgb(image):
    from PIL import Image

    if not isinstance(image, Image.Image):
        return image
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _resize(image, size, resample, data_format=None):
    from PIL import Image

    if isinstance(image, np.ndarray):
        h, w = size
        pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
        pil_img = pil_img.resize((w, h), resample=resample)
        return np.array(pil_img)
    return image


def _to_channel_first(image, input_format):
    if input_format == ChannelDimension.FIRST:
        return image
    if input_format == ChannelDimension.LAST:
        return np.transpose(image, (2, 0, 1))
    return image


class Gemma4ImageProcessor(BaseImageProcessor):
    """Image processor for Gemma 4.

    Aspect-ratio preserving resize, rescale to [0,1], output as channels-first.
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

    def aspect_ratio_preserving_resize(self, image, patch_size, max_patches, pooling_kernel_size, input_data_format):
        if input_data_format == ChannelDimension.FIRST:
            height, width = image.shape[1], image.shape[2]
        else:
            height, width = image.shape[0], image.shape[1]

        target_px = max_patches * (patch_size ** 2)
        factor = math.sqrt(target_px / (height * width))
        side_mult = pooling_kernel_size * patch_size

        target_height = int(math.floor(factor * height / side_mult)) * side_mult
        target_width = int(math.floor(factor * width / side_mult)) * side_mult

        if target_height == 0 and target_width == 0:
            raise ValueError("Attempting to resize to a 0 x 0 image.")

        max_side_length = (max_patches // pooling_kernel_size ** 2) * side_mult
        if target_height == 0:
            target_height = side_mult
            target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
        elif target_width == 0:
            target_width = side_mult
            target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

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

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        patch_size = kwargs.get("patch_size", self.patch_size)
        max_soft_tokens = kwargs.get("max_soft_tokens", self.max_soft_tokens)
        pooling_kernel_size = kwargs.get("pooling_kernel_size", self.pooling_kernel_size)
        max_patches = max_soft_tokens * pooling_kernel_size ** 2

        images = self.fetch_images(images)
        images = make_list_of_images(images)

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
                    image, patch_size, max_patches, pooling_kernel_size, input_data_format
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
            num_soft_tokens_per_image.append(num_patches // (pooling_kernel_size ** 2))

        shapes = {img.shape for img in processed}
        if len(shapes) > 1:
            data = {"pixel_values": processed}
        else:
            data = {"pixel_values": np.stack(processed)}

        result = BatchFeature(data=data, tensor_type=return_tensors)
        result["num_soft_tokens_per_image"] = num_soft_tokens_per_image
        return result

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)


class Gemma4Processor(ProcessorMixin):
    """Combined processor for Gemma 4 (image + text)."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Gemma4ImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 280,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = Gemma4ImageProcessor()

        self.image_seq_length = image_seq_length
        self.image_token_id = getattr(tokenizer, "image_token_id", None)
        self.boi_token = getattr(tokenizer, "boi_token", "")
        self.eoi_token = getattr(tokenizer, "eoi_token", "")
        self.image_token = getattr(tokenizer, "image_token", "")

        image_tokens_expanded = self.image_token * image_seq_length
        self.full_image_sequence = f"{self.boi_token}{image_tokens_expanded}{self.eoi_token}"

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        if isinstance(text, str):
            text = [text]

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            image_inputs = self.image_processor(images)

            num_soft_tokens = image_inputs.pop("num_soft_tokens_per_image", None)

            if text is not None and num_soft_tokens is not None:
                replacements = [
                    (f"\n\n{self.boi_token}" + self.image_token * n + f"{self.eoi_token}\n\n")
                    for n in num_soft_tokens
                ]
                replacements_iter = iter(replacements)
                pattern = re.escape(self.image_token)
                text = [
                    re.sub(pattern, lambda _: next(replacements_iter), prompt)
                    for prompt in text
                ]
            elif text is not None:
                text = [
                    prompt.replace(self.image_token, self.full_image_sequence)
                    for prompt in text
                ]

        return_tensors = kwargs.pop("return_tensors", None)
        add_special_tokens = kwargs.pop("add_special_tokens", True)
        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
            )

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = list(self.tokenizer.model_input_names)
        names.extend(self.image_processor.model_input_names)
        return list(dict.fromkeys(names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Load image processor config from processor_config.json
        ip_config = {}
        try:
            import json

            if is_local:
                config_path = model_path / "processor_config.json"
            else:
                from huggingface_hub import hf_hub_download

                config_path = Path(
                    hf_hub_download(pretrained_model_name_or_path, "processor_config.json")
                )
            if config_path.exists():
                with open(config_path) as f:
                    proc_config = json.load(f)
                ip_config = proc_config.get("image_processor", {})
                ip_config.pop("image_processor_type", None)
        except Exception:
            pass

        image_processor = Gemma4ImageProcessor(**ip_config)

        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            try:
                if is_local:
                    jinja_path = model_path / "chat_template.jinja"
                else:
                    from huggingface_hub import hf_hub_download

                    jinja_path = Path(
                        hf_hub_download(pretrained_model_name_or_path, "chat_template.jinja")
                    )
                if jinja_path.exists():
                    chat_template = jinja_path.read_text(encoding="utf-8")
                    tokenizer.chat_template = chat_template
            except Exception:
                pass

        image_seq_length = ip_config.get("max_soft_tokens", 280)

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
        )


# Register with AutoProcessor
install_auto_processor_patch("gemma4", Gemma4Processor)
