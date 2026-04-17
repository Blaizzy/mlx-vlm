"""
Processor class for Llama4.

Adapted from HuggingFace Transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/processing_llama4.py
"""

from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx

# Supported aspect ratios for Llama4 tiling
POSSIBLE_RESOLUTIONS = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (3, 1),
    (4, 1),
]


class Llama4ImageProcessor(BaseImageProcessor):
    """Minimal image processor for Llama4 (replaces torchvision-dependent fast processor)."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size=336,
        max_patches=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.max_patches = max_patches
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def _get_best_resolution(self, image_width, image_height):
        best = (1, 1)
        best_fit = float("inf")
        for h, w in POSSIBLE_RESOLUTIONS:
            if h * w > self.max_patches:
                continue
            scale = min(w * self.size / image_width, h * self.size / image_height)
            fit = abs(image_width * scale - w * self.size) + abs(
                image_height * scale - h * self.size
            )
            if fit < best_fit:
                best_fit = fit
                best = (h, w)
        return best

    def _split_into_tiles(self, image, aspect_ratio):
        ratio_h, ratio_w = aspect_ratio
        target_w = ratio_w * self.size
        target_h = ratio_h * self.size
        image = image.resize((target_w, target_h), Image.BICUBIC)
        tiles = []
        for i in range(ratio_h):
            for j in range(ratio_w):
                box = (
                    j * self.size,
                    i * self.size,
                    (j + 1) * self.size,
                    (i + 1) * self.size,
                )
                tiles.append(image.crop(box))
        return tiles

    def _preprocess_single(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Normalize: rescale to [0,1] then normalize
        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        # HWC -> CHW
        return arr.transpose(2, 0, 1)

    def preprocess(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]

        all_pixel_values = []
        aspect_ratios = []

        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))
            ar = self._get_best_resolution(img.width, img.height)
            aspect_ratios.append(ar)
            tiles = self._split_into_tiles(img, ar)
            for tile in tiles:
                all_pixel_values.append(self._preprocess_single(tile))

        pixel_values = np.stack(all_pixel_values)
        return {
            "pixel_values": pixel_values,
            "aspect_ratios": aspect_ratios,
        }

    def fetch_images(self, images):
        """Normalize images input to a flat list of PIL Images."""
        if isinstance(images, Image.Image):
            return [images]
        if isinstance(images, (list, tuple)):
            result = []
            for img in images:
                if isinstance(img, Image.Image):
                    result.append(img)
                elif isinstance(img, (list, tuple)):
                    result.extend(img)
                else:
                    result.append(img)
            return result
        return [images]


class Llama4Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        pixel_shuffle_ratio: float = 0.5,
        fake_image_token="<|image|>",
        image_token="<|image|>",
        start_of_image_token="<|image_start|>",
        end_of_image_token="<|image_end|>",
        patch_token="<|patch|>",
        tile_x_separator_token="<|tile_x_separator|>",
        tile_y_separator_token="<|tile_y_separator|>",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        self.patch_size = patch_size

        self.fake_image_token = fake_image_token
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.start_of_img_token = start_of_image_token
        self.end_of_img_token = end_of_image_token
        self.img_patch_token = patch_token
        self.tile_token = tile_x_separator_token
        self.tile_global_token = tile_y_separator_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        pp_cfg = Path(pretrained_model_name_or_path) / "preprocessor_config.json"
        ip_kwargs = {}
        if pp_cfg.exists():
            with open(pp_cfg) as f:
                cfg = json.load(f)
            size = cfg.get("size", {})
            ip_kwargs["size"] = (
                size.get("height", 336) if isinstance(size, dict) else size
            )
            ip_kwargs["max_patches"] = cfg.get("max_patches", 16)
            ip_kwargs["image_mean"] = cfg.get("image_mean", (0.5, 0.5, 0.5))
            ip_kwargs["image_std"] = cfg.get("image_std", (0.5, 0.5, 0.5))

        image_processor = Llama4ImageProcessor(**ip_kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def _prompt_split_image(self, aspect_ratio, num_patches_per_chunk):
        """Create a structured string representation of image tokens."""
        img_string = "<|image_start|>"
        ratio_h, ratio_w = aspect_ratio
        if ratio_h * ratio_w > 1:
            for yy in range(ratio_h):
                for xx in range(ratio_w):
                    img_string += "<|patch|>" * num_patches_per_chunk
                    if xx < ratio_w - 1:
                        img_string += "<|tile_x_separator|>"
                img_string += "<|tile_y_separator|>"

        img_string += "<|image|>"
        img_string += "<|patch|>" * num_patches_per_chunk
        img_string += "<|image_end|>"
        return img_string

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
        **kwargs,
    ) -> BatchFeature:
        kwargs.pop("padding", None)
        kwargs.pop("return_tensors", None)

        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images
        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images)
            image_height, image_width = image_inputs["pixel_values"][0].shape[-2:]
            num_patches_per_chunk = int(
                (image_height // self.patch_size)
                * (image_width // self.patch_size)
                // self.downsample_ratio
            )
            aspect_ratios = image_inputs.pop("aspect_ratios")

            total_placeholders = sum(
                prompt.count(self.fake_image_token) for prompt in text
            )
            if total_placeholders != len(images):
                raise ValueError(
                    f"Found {total_placeholders} placeholders across the batch, "
                    f"but have {len(images)} flattened images."
                )

            image_index = 0
            processed_text = []
            for prompt in text:
                placeholder_count = prompt.count(self.fake_image_token)
                if placeholder_count == 0:
                    processed_text.append(prompt)
                    continue
                prompt_splits = prompt.split(self.fake_image_token)
                new_prompt = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_prompt.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = self._prompt_split_image(
                            aspect_ratios[image_index],
                            num_patches_per_chunk,
                        )
                        image_index += 1
                        new_prompt.append(tokens_for_this_image)
                processed_text.append("".join(new_prompt))

            if image_index != len(images):
                raise ValueError(
                    "Number of image placeholders in the prompt does not match the number of images."
                )

            text = processed_text

        return_tensors = kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **kwargs)

        return BatchFeature(data=to_mlx({**text_inputs, **image_inputs}))

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Llama4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch("llama4", Llama4Processor)
