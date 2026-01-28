"""
MLX-based Molmo Processor.

This module provides an MLX-native processor for Molmo models that doesn't
require torch, torchvision, or tensorflow.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.processing_utils import ProcessorMixin

# CLIP normalization constants
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pad_to_bounding_box(
    image: np.ndarray,
    offset_height: int,
    offset_width: int,
    target_height: int,
    target_width: int,
    value: int = 0,
) -> np.ndarray:
    """Pad image to target bounding box."""
    height, width = image.shape[:2]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    if image.ndim == 3:
        padding = [
            [offset_height, after_padding_height],
            [offset_width, after_padding_width],
            [0, 0],
        ]
    else:
        padding = [
            [offset_height, after_padding_height],
            [offset_width, after_padding_width],
        ]
    return np.pad(image, padding, constant_values=value)


def normalize_image(
    image: np.ndarray, offset: Tuple[float, ...], scale: Tuple[float, ...]
) -> np.ndarray:
    """Normalize image with mean and std."""
    image = image.astype(np.float32)
    image -= np.array(offset, dtype=np.float32)[None, None, :]
    image /= np.array(scale, dtype=np.float32)[None, None, :]
    return image


def resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
    pad_value: float = 0,
    normalize: bool = True,
    image_mean: Tuple[float, ...] = OPENAI_CLIP_MEAN,
    image_std: Tuple[float, ...] = OPENAI_CLIP_STD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and pad image using PIL (no torch/tensorflow)."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Calculate scale
    image_scale_y = np.float32(desired_height) / np.float32(height)
    image_scale_x = np.float32(desired_width) / np.float32(width)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.float32(height) * image_scale)
    scaled_width = int(np.float32(width) * image_scale)

    # Use PIL for resizing (bilinear interpolation)
    pil_image = Image.fromarray(
        (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    )
    pil_image = pil_image.resize(
        (scaled_width, scaled_height), Image.Resampling.BILINEAR
    )
    image = np.array(pil_image).astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)

    # Pad to desired size
    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0],
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)

    if normalize:
        image = normalize_image(image, offset=image_mean, scale=image_std)

    return image, image_mask


def select_tiling(
    h: int, w: int, patch_size: int, max_num_patches: int
) -> Tuple[int, int]:
    """Select best tiling for image."""
    tilings = []
    for i in range(1, max_num_patches + 1):
        for j in range(1, max_num_patches + 1):
            if i * j <= max_num_patches:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([h, w], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)

    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)

    return tuple(candidate_tilings[ix])


def rearrange_patches(
    patches: np.ndarray, dh: int, dw: int, h: int, w: int
) -> np.ndarray:
    """Rearrange patches: 'p (h dh) (w dw) c -> p (h w) (dh dw c)'"""
    p, H, W, c = patches.shape
    patches = patches.reshape(p, h, dh, w, dw, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(p, h * w, dh * dw * c)
    return patches


def rearrange_mask(mask: np.ndarray, dh: int, dw: int, h: int, w: int) -> np.ndarray:
    """Rearrange mask: 'p (h dh) (w dw) -> p (h w) (dh dw)'"""
    p, H, W = mask.shape
    mask = mask.reshape(p, h, dh, w, dw)
    mask = mask.transpose(0, 1, 3, 2, 4)
    mask = mask.reshape(p, h * w, dh * dw)
    return mask


def rearrange_global(image: np.ndarray, dh: int, dw: int, h: int, w: int) -> np.ndarray:
    """Rearrange global image: '(h dh) (w dw) c -> (h w) (dh dw c)'"""
    H, W, c = image.shape
    image = image.reshape(h, dh, w, dw, c)
    image = image.transpose(0, 2, 1, 3, 4)
    image = image.reshape(h * w, dh * dw * c)
    return image


class MolmoImageProcessor(BaseImageProcessor):
    """MLX-based image processor for Molmo."""

    model_input_names = ["images", "image_input_idx", "image_masks"]

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margins: List[int] = None,
        base_image_input_size: List[int] = None,
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins or [4, 4]
        self.base_image_input_size = base_image_input_size or [336, 336]
        self.image_token_length_w = image_token_length_w
        self.image_token_length_h = image_token_length_h
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = tuple(image_mean) if image_mean else OPENAI_CLIP_MEAN
        self.image_std = tuple(image_std) if image_std else OPENAI_CLIP_STD

    def image_to_patches_and_tokens(
        self,
        image: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert image to patches and tokens."""
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size
        tokens_per_image = self.image_token_length_w * self.image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        left_margin, right_margin = self.overlap_margins
        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            self.max_crops,
        )

        src, img_mask = resize_and_pad(
            image,
            [
                tiling[0] * crop_window_size + total_margin_pixels,
                tiling[1] * crop_window_size + total_margin_pixels,
            ],
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

        patches_arr = []
        mask_arr = []
        patch_ordering_arr = []

        on = 0
        for i in range(tiling[0]):
            y0 = i * crop_window_size
            crop_y0 = 0 if i == 0 else left_margin // 2

            crop_h = image_base_patch_h - (right_margin + left_margin)
            if i == 0:
                crop_h += left_margin
            if i == (tiling[0] - 1):
                crop_h += right_margin

            for j in range(tiling[1]):
                x0 = j * crop_window_size
                crop_x0 = 0 if j == 0 else left_margin // 2

                crop_w = image_base_patch_w - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1] - 1):
                    crop_w += right_margin

                pooled_w = (crop_w + 1) // 2
                pooled_h = (crop_h + 1) // 2

                ordering = np.reshape(
                    np.arange(on, on + pooled_h * pooled_w, dtype=np.int32),
                    (pooled_h, pooled_w, 1),
                )
                patch_ordering_arr.append(
                    pad_to_bounding_box(
                        ordering,
                        crop_y0,
                        crop_x0,
                        self.image_token_length_h,
                        self.image_token_length_w,
                        value=-1,
                    )[:, :, 0]
                )
                patches_arr.append(src[y0 : y0 + crop_size, x0 : x0 + crop_size])
                mask_arr.append(img_mask[y0 : y0 + crop_size, x0 : x0 + crop_size])

                on += pooled_h * pooled_w

        patches = np.stack(patches_arr)
        patch_ordering = np.stack(patch_ordering_arr)
        img_mask = np.stack(mask_arr)

        # Rearrange patches
        patches = rearrange_patches(
            patches,
            base_image_input_d,
            base_image_input_d,
            image_base_patch_h,
            image_base_patch_w,
        )
        img_mask = rearrange_mask(
            img_mask,
            base_image_input_d,
            base_image_input_d,
            image_base_patch_h,
            image_base_patch_w,
        )

        img_mask = img_mask.astype(np.float32).mean(axis=-1)
        patch_ordering = np.reshape(patch_ordering, [-1])
        valid = patch_ordering >= 0

        # Transpose order
        patch_ordering_rh = np.reshape(
            patch_ordering,
            [
                tiling[0],
                tiling[1],
                self.image_token_length_h,
                self.image_token_length_w,
            ],
        )
        patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
        patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

        patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

        # Build output tokens
        h = tiling[0] * crop_window_patches + (right_margin + left_margin)
        w = tiling[1] * crop_window_patches + (right_margin + left_margin)
        per_row = np.full(((w + 1) // 2,), image_patch_token_id)
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)

        joint = np.tile(per_row, [(h + 1) // 2])
        joint = [[image_start_token_id], joint, [image_end_token_id]]

        # Global image
        resized, _ = resize_and_pad(
            image,
            base_image_input_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        resized = rearrange_global(
            resized,
            base_image_input_d,
            base_image_input_d,
            image_base_patch_h,
            image_base_patch_w,
        )
        patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

        patch_ordering = np.where(
            patch_ordering >= 0, patch_ordering + tokens_per_image, -1
        )
        patch_ordering = np.concatenate(
            [np.arange(0, tokens_per_image), patch_ordering], 0
        )

        per_row = np.full((self.image_token_length_w,), image_patch_token_id)
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)
        extra_tokens = np.tile(per_row, [self.image_token_length_h])
        joint = [
            [image_start_token_id],
            extra_tokens,
            [image_end_token_id],
        ] + joint

        joint = np.concatenate(joint, 0)
        img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)

        return patches, joint, patch_ordering, img_mask

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
        image_patch_token_id: int,
    ) -> np.ndarray:
        """Build image input indices."""
        tokens_per_image = self.image_token_length_w * self.image_token_length_h

        image_input_idx = image_tokens == image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        if patch_order is not None:
            n_tokens = image_input_idx.shape[0]
            patch_order = np.reshape(patch_order, [-1])

            valid = patch_order >= 0
            n_valid_patches = valid.sum()

            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(
                n_valid_patches, dtype=np.int32
            )

            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            valid_int = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex * valid_int]
            image_input_idx = image_input_idx * valid_int - 100 * (1 - valid_int)
            image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])

        return image_input_idx

    def preprocess(
        self,
        image: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess a single image."""
        crops, image_tokens, patch_ordering, img_mask = (
            self.image_to_patches_and_tokens(
                image,
                image_patch_token_id,
                image_col_token_id,
                image_start_token_id,
                image_end_token_id,
            )
        )
        patch_idx = self.build_image_input_idx(
            image_tokens, patch_ordering, image_patch_token_id
        )
        return crops, image_tokens, patch_idx, img_mask


class MolmoProcessor(ProcessorMixin):
    """MLX-based processor for Molmo."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MolmoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = MolmoImageProcessor()
        # Molmo uses these specific token names
        self.image_patch_token = "<im_patch>"
        self.image_col_token = "<im_col>"
        self.image_start_token = "<im_start>"
        self.image_end_token = "<im_end>"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[str, List[str]] = None,
        **kwargs,
    ) -> BatchFeature:
        """Process images and text for Molmo."""
        if images is None and text is None:
            raise ValueError("You must provide either images or text.")

        # Get token IDs
        image_patch_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_patch_token
        )
        image_col_token_id = self.tokenizer.convert_tokens_to_ids(self.image_col_token)
        image_start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_start_token
        )
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)

        # Validate token IDs
        if image_patch_token_id is None:
            raise ValueError(
                f"Token '{self.image_patch_token}' not found in tokenizer vocabulary"
            )
        if image_col_token_id is None:
            raise ValueError(
                f"Token '{self.image_col_token}' not found in tokenizer vocabulary"
            )
        if image_start_token_id is None:
            raise ValueError(
                f"Token '{self.image_start_token}' not found in tokenizer vocabulary"
            )
        if image_end_token_id is None:
            raise ValueError(
                f"Token '{self.image_end_token}' not found in tokenizer vocabulary"
            )

        # Process images
        if images is not None:
            images = make_list_of_images(images)
            # Convert PIL images to numpy arrays
            np_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                    np_images.append(np.array(img).astype(np.float32) / 255.0)
                elif isinstance(img, np.ndarray):
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                    np_images.append(img)
                else:
                    np_images.append(np.array(img).astype(np.float32) / 255.0)
            images = np_images

        # Tokenize text
        if text is not None:
            if isinstance(text, str):
                text = [text]
            tokens_list = [self.tokenizer.encode(t) for t in text]
        else:
            tokens_list = [[]]

        # Process each image with text
        if images is not None and len(images) > 0:
            all_crops = []
            all_image_idx = []
            all_masks = []
            all_input_ids = []

            for i, (img, tokens) in enumerate(
                zip(
                    images,
                    (
                        tokens_list
                        if len(tokens_list) == len(images)
                        else [tokens_list[0]] * len(images)
                    ),
                )
            ):
                crops, image_tokens, patch_idx, img_mask = (
                    self.image_processor.preprocess(
                        img,
                        image_patch_token_id,
                        image_col_token_id,
                        image_start_token_id,
                        image_end_token_id,
                    )
                )

                # Combine image tokens with text tokens
                combined_tokens = np.concatenate([image_tokens, np.array(tokens)])

                # Adjust patch_idx for the position in combined tokens
                all_crops.append(crops)
                all_image_idx.append(patch_idx)
                all_masks.append(img_mask)
                all_input_ids.append(combined_tokens)

            # Stack results
            pixel_values = mx.array(
                np.concatenate(all_crops, axis=0).astype(np.float32)
            )
            image_input_idx = mx.array(
                np.concatenate(all_image_idx, axis=0).astype(np.int32)
            )
            image_masks = mx.array(np.concatenate(all_masks, axis=0).astype(np.float32))

            # Pad input_ids to same length
            max_len = max(len(ids) for ids in all_input_ids)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0
            padded_ids = []
            for ids in all_input_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    ids = np.pad(ids, (0, pad_len), constant_values=pad_token_id)
                padded_ids.append(ids.astype(np.int32))

            input_ids = mx.array(np.stack(padded_ids).astype(np.int32))

            return BatchFeature(
                data={
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "image_input_idx": image_input_idx,
                    "image_masks": image_masks,
                }
            )
        else:
            # Text only
            max_len = max(len(t) for t in tokens_list)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0
            padded = []
            for t in tokens_list:
                pad_len = max_len - len(t)
                if pad_len > 0:
                    t = t + [pad_token_id] * pad_len
                padded.append(t)

            return BatchFeature(data={"input_ids": mx.array(padded, dtype=mx.int32)})

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        """Apply chat template."""
        if chat_template is None:
            chat_template = self.chat_template
        if chat_template is None:
            chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template is None:
            # Default Molmo chat template
            chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "User: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}"
                "Assistant: {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}Assistant: {% endif %}"
            )

        from jinja2 import Environment

        # Use Environment with loopcontrols extension to support {% continue %} and {% break %}
        env = Environment(extensions=["jinja2.ext.loopcontrols"])
        template = env.from_string(chat_template)
        rendered = template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.tokenizer.encode(rendered)
        return rendered

    @property
    def model_input_names(self):
        """Get model input names."""
        return ["input_ids", "pixel_values", "image_input_idx", "image_masks"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model."""
        from huggingface_hub import hf_hub_download

        kwargs.pop("trust_remote_code", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        # Load image processor config
        image_processor_config = {}
        try:
            if is_local:
                config_path = model_path / "preprocessor_config.json"
            else:
                config_path = Path(
                    hf_hub_download(
                        pretrained_model_name_or_path, "preprocessor_config.json"
                    )
                )
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key in [
                    "max_crops",
                    "overlap_margins",
                    "base_image_input_size",
                    "image_token_length_w",
                    "image_token_length_h",
                    "image_patch_size",
                    "image_padding_mask",
                    "do_normalize",
                    "image_mean",
                    "image_std",
                ]:
                    if key in config:
                        image_processor_config[key] = config[key]
        except Exception:
            pass

        image_processor = MolmoImageProcessor(**image_processor_config)

        # Load chat template
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            try:
                if is_local:
                    jinja_path = model_path / "chat_template.jinja"
                else:
                    jinja_path = Path(
                        hf_hub_download(
                            pretrained_model_name_or_path, "chat_template.jinja"
                        )
                    )
                if jinja_path.exists():
                    chat_template = jinja_path.read_text(encoding="utf-8")
                    tokenizer.chat_template = chat_template
            except Exception:
                pass

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )


# Patch AutoProcessor for Molmo models
from transformers import AutoProcessor

_original_auto_processor_from_pretrained = AutoProcessor.from_pretrained


@classmethod
def _patched_auto_processor_from_pretrained(
    cls, pretrained_model_name_or_path, **kwargs
):
    """Patched from_pretrained that returns MolmoProcessor for molmo models."""
    from huggingface_hub import hf_hub_download

    model_path = Path(pretrained_model_name_or_path)
    is_local = model_path.exists() and model_path.is_dir()

    # Check if this is a molmo model
    is_molmo = False
    try:
        if is_local:
            config_path = model_path / "config.json"
        else:
            config_path = Path(
                hf_hub_download(pretrained_model_name_or_path, "config.json")
            )
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()

        is_molmo = model_type == "molmo"
    except Exception:
        pass

    if is_molmo:
        return MolmoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

    return _original_auto_processor_from_pretrained.__func__(
        cls, pretrained_model_name_or_path, **kwargs
    )


AutoProcessor.from_pretrained = _patched_auto_processor_from_pretrained
