import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_chw(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected an image with three dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.transpose(2, 0, 1)


class CohereCompassImageProcessor(ImageProcessingMixin):
    """Torch-free native-resolution image processor for Compass checkpoints."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        config=None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 16384,
        max_pixels: int = 3868706,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        if isinstance(config, dict):
            vision_config = config.get("vision_config") or {}
            patch_size = vision_config.get("patch_size", patch_size)
            temporal_patch_size = vision_config.get(
                "temporal_patch_size", temporal_patch_size
            )
            merge_size = vision_config.get("spatial_merge_size", merge_size)
            min_pixels = config.get("min_pixels") or min_pixels
            max_pixels = config.get("max_pixels") or max_pixels
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def fetch_images(self, images):
        return images if isinstance(images, list) else [images]

    def __call__(self, images: ImageInput, **kwargs):
        images = self.fetch_images(images)
        min_pixels = kwargs.get("min_pixels", self.min_pixels)
        max_pixels = kwargs.get("max_pixels", self.max_pixels)
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get(
            "temporal_patch_size", self.temporal_patch_size
        )
        merge_size = kwargs.get("merge_size", self.merge_size)
        do_resize = kwargs.get("do_resize", self.do_resize)
        all_patches = []
        all_grids = []

        for image in images:
            image = _to_chw(image)
            _, height, width = image.shape
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                pil = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
                pil = pil.resize(
                    (resized_width, resized_height), Image.Resampling.BICUBIC
                )
                image = np.asarray(pil).transpose(2, 0, 1)
            else:
                resized_height, resized_width = height, width

            image = image.astype(np.float32)
            if kwargs.get("do_rescale", self.do_rescale):
                image = image * kwargs.get("rescale_factor", self.rescale_factor)
            if kwargs.get("do_normalize", self.do_normalize):
                mean = np.asarray(kwargs.get("image_mean", self.image_mean))[
                    :, None, None
                ]
                std = np.asarray(kwargs.get("image_std", self.image_std))[:, None, None]
                image = (image - mean) / std

            patches = np.repeat(image[None, None], temporal_patch_size, axis=1)
            grid_t = 1
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            channels = patches.shape[2]
            patches = patches.reshape(
                1,
                grid_t,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
            all_patches.append(patches)
            all_grids.append([grid_t, grid_h, grid_w])

        data = {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.asarray(all_grids, dtype=np.int64),
        }
        return BatchFeature(data=to_mlx(data), tensor_type=kwargs.get("return_tensors"))

    preprocess = __call__

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        h, w = smart_resize(
            height,
            width,
            patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (h // patch_size) * (w // patch_size)


class CohereCompassProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        self.image_token = getattr(tokenizer, "image_token", "<|IMAGE_PAD|>")
        self.image_token_id = getattr(
            tokenizer,
            "image_token_id",
            tokenizer.convert_tokens_to_ids(self.image_token),
        )
        self.vision_start_token = getattr(
            tokenizer, "vision_start_token", "<|VISION_START|>"
        )
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|VISION_END|>")
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, **kwargs):
        if text is None:
            raise ValueError("You have to specify text.")
        text = text.copy() if isinstance(text, list) else [text]
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **kwargs)
            grid = image_inputs["image_grid_thw"]
            image_index = 0
            merge_length = self.image_processor.merge_size**2
            for i, prompt in enumerate(text):
                while self.image_token in prompt:
                    if image_index >= len(grid):
                        raise ValueError("More image placeholders than images")
                    count = int(grid[image_index].prod().item()) // merge_length
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * count, 1
                    )
                    prompt = text[i]
                    image_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)
            if image_index != len(grid):
                raise ValueError("More images than image placeholders")

        return_tensors = kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **kwargs)
        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs}), tensor_type=return_tensors
        )

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
            )
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_kwargs = dict(kwargs)
        tokenizer_kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        config = {}
        preprocessor_config = {}
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            config_path = path / "config.json"
            processor_path = path / "preprocessor_config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
            if processor_path.exists():
                preprocessor_config = json.loads(processor_path.read_text())
        else:
            from huggingface_hub import hf_hub_download

            for name, target in (
                ("config.json", "config"),
                ("preprocessor_config.json", "preprocessor_config"),
            ):
                try:
                    downloaded = hf_hub_download(pretrained_model_name_or_path, name)
                    value = json.loads(Path(downloaded).read_text())
                    if target == "config":
                        config = value
                    else:
                        preprocessor_config = value
                except Exception:
                    pass

            try:
                chat_template_path = hf_hub_download(
                    pretrained_model_name_or_path, "chat_template.jinja"
                )
                tokenizer.chat_template = Path(chat_template_path).read_text()
            except Exception:
                pass

        image_kwargs = {
            key: value
            for key, value in preprocessor_config.items()
            if key
            in {
                "patch_size",
                "temporal_patch_size",
                "merge_size",
                "min_pixels",
                "max_pixels",
                "do_resize",
                "do_rescale",
                "rescale_factor",
                "do_normalize",
                "image_mean",
                "image_std",
                "do_convert_rgb",
            }
        }
        image_processor = CohereCompassImageProcessor(config=config, **image_kwargs)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


install_auto_processor_patch("cohere_compass", CohereCompassProcessor)
