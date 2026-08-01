"""
Processor class for Kimi K3.

Adapted from the custom HuggingFace Transformers implementation:
https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_processor.py
https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_vision_processing.py

The K3 chat template renders one <|media_pad|> per image which the reference
model expands to the image's merged-token count at embedding time; this
processor pre-expands the placeholders at the prompt level instead so the
model-side merge is 1:1.
"""

import inspect
import json
import math
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import make_list_of_images, valid_images
from transformers.processing_utils import ProcessorMixin

IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
MEDIA_PAD = "<|media_pad|>"

# prompt_utils selects a chat renderer only when chat_template is non-None.
_CHAT_TEMPLATE_SENTINEL = "<kimi-k3-native-python-chat-renderer>"


class KimiK3ImageProcessor(BaseImageProcessor):
    """Image processor for Kimi K3 using the navit resize/pad/patchify pipeline."""

    model_input_names = ["pixel_values", "grid_thws"]

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        in_patch_limit: int = 65536,
        merge_kernel_size: List[int] = None,
        patch_limit_on_one_side: int = 512,
        transparent_bg_config: dict = None,
        transparent_bg_fill_stage: str = "before_resize",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.in_patch_limit = in_patch_limit
        self.merge_kernel_size = (
            merge_kernel_size if merge_kernel_size is not None else [2, 2]
        )
        self.patch_limit_on_one_side = patch_limit_on_one_side
        self.transparent_bg_config = transparent_bg_config
        self.transparent_bg_fill_stage = transparent_bg_fill_stage

    def _fill_transparent_bg(self, image: Image.Image) -> Image.Image:
        cfg = self.transparent_bg_config
        if cfg is None:
            return image.convert("RGB")
        if image.mode == "RGB":
            return image
        if "A" not in image.getbands() and "transparency" not in image.info:
            return image.convert("RGB")

        img = np.asarray(image.convert("RGBA"))
        h, w = img.shape[:2]
        pattern = cfg.get("pattern", "black")
        if pattern == "white":
            bg = np.full((h, w, 3), 255, dtype=np.uint8)
        elif pattern == "black":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
        elif pattern == "gray":
            bg = np.full((h, w, 3), 128, dtype=np.uint8)
        elif pattern == "chessboard":
            square = cfg.get("chessboard_square_size", 16)
            on_top_left = cfg.get("chessboard_square_on_top_left", True)
            white = cfg.get("chessboard_white_value", 255)
            gray = cfg.get("chessboard_gray_value", 200)
            bg = np.full((h, w, 3), white, dtype=np.uint8)
            for y in range(0, h, square):
                for x in range(0, w, square):
                    if (y // square + x // square) % 2 == (1 if on_top_left else 0):
                        bg[y : y + square, x : x + square] = gray
        else:
            raise ValueError(f"Invalid background pattern: {pattern}")

        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        out = alpha * img[:, :, :3] + (1 - alpha) * bg
        return Image.fromarray(out.astype(np.uint8))

    def rescale(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        patch_size = self.patch_size
        merge_h, merge_w = self.merge_kernel_size

        s1 = math.sqrt(
            self.in_patch_limit
            / (max(1.0, w // patch_size) * max(1.0, h // patch_size))
        )
        s2 = self.patch_limit_on_one_side * patch_size / w
        s3 = self.patch_limit_on_one_side * patch_size / h
        scale = min(1.0, s1, s2, s3)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        new_w = min(new_w, self.patch_limit_on_one_side * patch_size)
        new_h = min(new_h, self.patch_limit_on_one_side * patch_size)

        factor_w = merge_w * patch_size
        factor_h = merge_h * patch_size
        pad_w = (factor_w - new_w % factor_w) % factor_w
        pad_h = (factor_h - new_h % factor_h) % factor_h

        if self.transparent_bg_fill_stage == "before_resize":
            image = self._fill_transparent_bg(image)
        elif image.mode == "P":
            image = image.convert("RGBA")
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        image = self._fill_transparent_bg(image)

        if pad_w > 0 or pad_h > 0:
            padded = Image.new("RGB", (new_w + pad_w, new_h + pad_h), (0, 0, 0))
            padded.paste(image, (0, 0))
            image = padded

        return image

    def _preprocess(self, image: Image.Image) -> Tuple[mx.array, Tuple[int, int, int]]:
        image = self.rescale(image)
        w, h = image.size
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.image_mean)) / np.array(self.image_std)

        p = self.patch_size
        gh, gw = h // p, w // p
        patches = arr.reshape(gh, p, gw, p, 3)
        patches = patches.transpose(0, 2, 4, 1, 3)
        patches = patches.reshape(-1, 3, p, p)
        return mx.array(patches, dtype=mx.float32), (1, gh, gw)

    def preprocess(self, images, return_tensors=None, **kwargs):
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type.")

        pixel_values_list = []
        grid_thws = []
        for image in images:
            patches, grid_thw = self._preprocess(image)
            pixel_values_list.append(patches)
            grid_thws.append(grid_thw)

        return BatchFeature(
            data={
                "pixel_values": mx.concatenate(pixel_values_list, axis=0),
                "grid_thws": mx.array(grid_thws),
            },
            tensor_type=return_tensors,
        )

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


class KimiK3Processor(ProcessorMixin):
    """MLX-based processor for Kimi K3."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "KimiK3ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = MEDIA_PAD
        if image_processor is None:
            image_processor = KimiK3ImageProcessor()
        super().__init__(image_processor, tokenizer, **kwargs)
        if (
            self.chat_template is None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            self.chat_template = _CHAT_TEMPLATE_SENTINEL

    def apply_chat_template(self, conversation, **kwargs):
        kwargs.pop("chat_template", None)
        try:
            parameters = inspect.signature(
                self.tokenizer.apply_chat_template
            ).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def make_image_prompt(self, width: int, height: int, num_tokens: int) -> str:
        return (
            f"<|media_begin|>image {width}x{height}"
            f"<|media_content|>{MEDIA_PAD * num_tokens}<|media_end|>"
        )

    def __call__(self, images=None, text=None, **kwargs):
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        image_prompts = []
        placeholder_counts = []
        if images is not None:
            image_list = make_list_of_images(images)
            image_inputs = self.image_processor(image_list)
            merge_length = (
                self.image_processor.merge_kernel_size[0]
                * self.image_processor.merge_kernel_size[1]
            )
            for image, grid_thw in zip(image_list, image_inputs["grid_thws"]):
                t, gh, gw = (int(v) for v in grid_thw)
                n = t * gh * gw // merge_length
                placeholder_counts.append(n)
                w, h = image.size if hasattr(image, "size") else (0, 0)
                image_prompts.append(self.make_image_prompt(w, h, n))
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError("Invalid input text. Please provide a string or list.")

        if text is not None:
            expanded_texts = []
            img_idx = 0
            for t in text:
                if IMAGE_PLACEHOLDER in t:
                    parts = t.split(IMAGE_PLACEHOLDER)
                    merged = parts[0]
                    for part in parts[1:]:
                        merged += image_prompts[img_idx] + part
                        img_idx += 1
                    t = merged
                elif placeholder_counts and MEDIA_PAD in t:
                    parts = t.split(MEDIA_PAD)
                    merged = parts[0]
                    for part in parts[1:]:
                        merged += MEDIA_PAD * placeholder_counts[img_idx] + part
                        img_idx += 1
                    t = merged
                expanded_texts.append(t)

            all_input_ids = [self.tokenizer.encode(t) for t in expanded_texts]

            max_len = max(len(ids) for ids in all_input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0

            padded_input_ids = []
            attention_masks = []
            for ids in all_input_ids:
                padding_length = max_len - len(ids)
                padded_input_ids.append(ids + [pad_token_id] * padding_length)
                attention_masks.append([1] * len(ids) + [0] * padding_length)

            text_inputs = {
                "input_ids": mx.array(padded_input_ids),
                "attention_mask": mx.array(attention_masks),
            }
        else:
            text_inputs = {}

        data = {**text_inputs, **image_inputs}
        if text is not None:
            data["image_token_id"] = int(
                self.tokenizer.convert_tokens_to_ids(self.image_token)
            )
        return BatchFeature(data=data)

    def save_pretrained(self, *args, **kwargs):
        if self.chat_template != _CHAT_TEMPLATE_SENTINEL:
            return super().save_pretrained(*args, **kwargs)
        self.chat_template = None
        try:
            return super().save_pretrained(*args, **kwargs)
        finally:
            self.chat_template = _CHAT_TEMPLATE_SENTINEL

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import hf_hub_download

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        image_processor_config = {}
        try:
            if is_local:
                preproc_path = model_path / "preprocessor_config.json"
            else:
                preproc_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "preprocessor_config.json"
                    )
                )
            if preproc_path.exists():
                with open(preproc_path, "r", encoding="utf-8") as f:
                    preproc_cfg = json.load(f)
                media_cfg = preproc_cfg.get("media_proc_cfg", {})
                for src, dst in (
                    ("patch_size", "patch_size"),
                    ("in_patch_limit", "in_patch_limit"),
                    ("patch_limit_on_one_side", "patch_limit_on_one_side"),
                ):
                    if src in media_cfg:
                        image_processor_config[dst] = media_cfg[src]
                if "image_mean" in media_cfg:
                    image_processor_config["image_mean"] = tuple(
                        media_cfg["image_mean"]
                    )
                if "image_std" in media_cfg:
                    image_processor_config["image_std"] = tuple(media_cfg["image_std"])
                if "merge_kernel_size" in media_cfg:
                    mks = media_cfg["merge_kernel_size"]
                    if isinstance(mks, int):
                        mks = [mks, mks]
                    image_processor_config["merge_kernel_size"] = mks
                if "transparent_bg_config" in media_cfg:
                    image_processor_config["transparent_bg_config"] = media_cfg[
                        "transparent_bg_config"
                    ]
                if "transparent_bg_fill_stage" in media_cfg:
                    image_processor_config["transparent_bg_fill_stage"] = media_cfg[
                        "transparent_bg_fill_stage"
                    ]
        except Exception:
            pass

        return cls(
            image_processor=KimiK3ImageProcessor(**image_processor_config),
            tokenizer=tokenizer,
        )


from ..base import install_auto_processor_patch

install_auto_processor_patch("kimi_k3", KimiK3Processor)
