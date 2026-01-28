"""Image processor for Jina VLM in MLX-VLM."""

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Default special token IDs
DEFAULT_PATCH_TOKEN_ID = 151665  # <im_patch>
DEFAULT_START_TOKEN_ID = 151666  # <im_start>
DEFAULT_END_TOKEN_ID = 151667  # <im_end>
DEFAULT_COLUMN_TOKEN_ID = 151668  # <im_col>


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Resize dimensions while maintaining aspect ratio and constraints."""
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


def patchify(array: np.ndarray, patch_size: int, batched: bool = False) -> np.ndarray:
    """Reshape image(s) to patches."""
    if len(array.shape) == 3 and not batched:
        h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = array.reshape(h_patches, patch_size, w_patches, patch_size, c)
        array = array.transpose(0, 2, 1, 3, 4)
        return array.reshape(h_patches * w_patches, patch_size * patch_size * c)
    elif len(array.shape) == 4 or (len(array.shape) == 3 and batched):
        if len(array.shape) == 3:
            bs, h, w = array.shape
            c = 1
            array = array[..., None]
        else:
            bs, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = array.reshape(bs, h_patches, patch_size, w_patches, patch_size, c)
        array = array.transpose(0, 1, 3, 2, 4, 5)
        result = array.reshape(bs, h_patches * w_patches, patch_size * patch_size * c)
        if c == 1:
            result = result[..., 0] if result.shape[-1] == 1 else result.mean(axis=-1)
        return result
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")


class ImageProcessor:
    """Image processor for Jina VLM (standalone, not a BaseImageProcessor)."""

    def __init__(
        self,
        config: Optional[dict] = None,
        base_input_size: Tuple[int, int] = (378, 378),
        patch_size: int = 14,
        max_crops: int = 12,
        min_pixels: int = 3136,
        max_pixels: int = 1003520,
        overlap_margins: Tuple[int, int] = (4, 4),
        pooling_h: int = 2,
        pooling_w: int = 2,
        use_column_tokens: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        patch_token_id: int = DEFAULT_PATCH_TOKEN_ID,
        start_token_id: int = DEFAULT_START_TOKEN_ID,
        end_token_id: int = DEFAULT_END_TOKEN_ID,
        column_token_id: int = DEFAULT_COLUMN_TOKEN_ID,
    ):
        self.base_input_size = base_input_size
        self.patch_size = patch_size
        self.max_crops = max_crops
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.overlap_margins = overlap_margins
        self.pooling_h = pooling_h
        self.pooling_w = pooling_w
        self.use_column_tokens = use_column_tokens
        self.image_mean = image_mean or CLIP_MEAN
        self.image_std = image_std or CLIP_STD

        self.patch_token_id = patch_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.column_token_id = column_token_id

        self.crop_patches = base_input_size[0] // patch_size
        self.token_length_h = (self.crop_patches + pooling_h - 1) // pooling_h
        self.token_length_w = (self.crop_patches + pooling_w - 1) // pooling_w
        self.tokens_per_image = self.token_length_h * self.token_length_w

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - 0.5) * 2.0

    def resize_image(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((size[1], size[0]), Image.BICUBIC)
        resized = np.array(pil_image, dtype=np.float32) / 255.0
        mask = np.ones((size[0], size[1]), dtype=np.bool_)
        return resized, mask

    def select_tiling(
        self,
        h: int,
        w: int,
        patch_size: int,
        max_crops: int,
    ) -> Tuple[int, int]:
        tilings = []
        for i in range(1, max_crops + 1):
            for j in range(1, max_crops + 1):
                if i * j <= max_crops:
                    tilings.append((i, j))

        tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
        candidate_tilings = np.array(tilings, dtype=np.int32)
        candidate_resolutions = candidate_tilings * patch_size

        original_size = np.array([h, w], dtype=np.float32)
        with np.errstate(divide="ignore"):
            required_scale = candidate_resolutions.astype(np.float32) / original_size
        required_scale = np.min(required_scale, axis=-1, keepdims=True)

        if np.all(required_scale < 1):
            ix = np.argmax(required_scale)
        else:
            required_scale = np.where(required_scale < 1.0, 1e9, required_scale)
            ix = np.argmin(required_scale)

        return tuple(candidate_tilings[ix])

    def _get_patches_from_tiling(
        self,
        num_tiles: int,
        pooling_size: int,
        crop_patches: int,
        crop_window_patches: int,
        left_margin: int,
        right_margin: int,
    ) -> int:
        if num_tiles > 1:
            left_crop = (
                (crop_window_patches + left_margin + pooling_size - 1)
                // pooling_size
                * pooling_size
            )
            middle_crop = (
                (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
            )
            right_crop = (
                (crop_window_patches + right_margin + pooling_size - 1)
                // pooling_size
                * pooling_size
            )
            return left_crop + (num_tiles - 2) * middle_crop + right_crop
        else:
            return (crop_patches + pooling_size - 1) // pooling_size * pooling_size

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: Optional[np.ndarray],
    ) -> np.ndarray:
        image_input_idx = image_tokens == self.patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        if patch_order is not None:
            patch_order = np.reshape(patch_order, [-1])
            valid = patch_order >= 0
            n_valid_patches = valid.sum()

            if len(image_input_idx) != n_valid_patches:
                raise ValueError(
                    f"Mismatch: {len(image_input_idx)} patch tokens but {n_valid_patches} valid patches"
                )

            sorted_patch_ixs = np.zeros([image_input_idx.shape[0]], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(
                n_valid_patches, dtype=np.int32
            )
            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            valid_mask = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex * valid_mask]
            image_input_idx = image_input_idx * valid_mask - 10000 * (1 - valid_mask)

        return np.reshape(image_input_idx, [-1, self.tokens_per_image])

    def crop_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_margin, right_margin = self.overlap_margins
        total_margin_pixels = self.patch_size * (right_margin + left_margin)
        crop_patches = self.crop_patches
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * self.patch_size

        original_h, original_w = image.shape[:2]

        tiling = self.select_tiling(
            original_h - total_margin_pixels,
            original_w - total_margin_pixels,
            crop_window_size,
            self.max_crops,
        )

        target_h = tiling[0] * crop_window_size + total_margin_pixels
        target_w = tiling[1] * crop_window_size + total_margin_pixels
        src, img_mask = self.resize_image(image, (target_h, target_w))
        src = self.normalize(src)

        patches_arr = []
        mask_arr = []
        patch_ordering_arr = []

        crop_size = self.base_input_size[0]
        on = 0

        for i in range(tiling[0]):
            y0 = i * crop_window_size
            crop_y0 = 0 if i == 0 else left_margin // self.pooling_h
            crop_h = crop_patches - (right_margin + left_margin)
            if i == 0:
                crop_h += left_margin
            if i == (tiling[0] - 1):
                crop_h += right_margin

            for j in range(tiling[1]):
                x0 = j * crop_window_size
                crop_x0 = 0 if j == 0 else left_margin // self.pooling_w
                crop_w = crop_patches - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1] - 1):
                    crop_w += right_margin

                pooled_w = (crop_w + self.pooling_w - 1) // self.pooling_w
                pooled_h = (crop_h + self.pooling_h - 1) // self.pooling_h
                after_padding_width = self.token_length_w - pooled_w - crop_x0
                after_padding_height = self.token_length_h - pooled_h - crop_y0

                patch_ordering_arr.append(
                    np.pad(
                        np.reshape(
                            np.arange(on, on + pooled_h * pooled_w, dtype=np.int32),
                            (pooled_h, pooled_w),
                        ),
                        [
                            [crop_y0, after_padding_height],
                            [crop_x0, after_padding_width],
                        ],
                        constant_values=-1,
                        mode="constant",
                    )
                )

                crop = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
                if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
                    padded = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
                    padded[: crop.shape[0], : crop.shape[1]] = crop
                    crop = padded
                patches_arr.append(crop)

                crop_mask = img_mask[y0 : y0 + crop_size, x0 : x0 + crop_size]
                if crop_mask.shape[0] < crop_size or crop_mask.shape[1] < crop_size:
                    padded_mask = np.zeros((crop_size, crop_size), dtype=np.bool_)
                    padded_mask[: crop_mask.shape[0], : crop_mask.shape[1]] = crop_mask
                    crop_mask = padded_mask
                mask_arr.append(crop_mask)

                on += pooled_h * pooled_w

        patches = np.stack(patches_arr)
        patch_ordering = np.stack(patch_ordering_arr)
        img_masks = np.stack(mask_arr)

        patches = patchify(patches, self.patch_size, batched=True)
        img_masks = patchify(
            img_masks.astype(np.float32), self.patch_size, batched=True
        )
        if img_masks.ndim == 3:
            img_masks = img_masks.mean(axis=-1)

        patch_ordering = np.reshape(patch_ordering, [-1])
        valid = patch_ordering >= 0

        patch_ordering_rh = np.reshape(
            patch_ordering,
            [tiling[0], tiling[1], self.token_length_h, self.token_length_w],
        )
        patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
        patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])
        patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

        h = self._get_patches_from_tiling(
            tiling[0],
            self.pooling_h,
            crop_patches,
            crop_window_patches,
            left_margin,
            right_margin,
        )
        w = self._get_patches_from_tiling(
            tiling[1],
            self.pooling_w,
            crop_patches,
            crop_window_patches,
            left_margin,
            right_margin,
        )

        per_row = np.full((w // self.pooling_w,), self.patch_token_id, dtype=np.int32)
        if self.use_column_tokens:
            per_row = np.concatenate([per_row, [self.column_token_id]], 0)
        joint = np.tile(per_row, [h // self.pooling_h])
        joint = [[self.start_token_id], joint, [self.end_token_id]]

        thumb, _ = self.resize_image(image, self.base_input_size)
        thumb = self.normalize(thumb)
        thumb_patches = patchify(thumb, self.patch_size, batched=False)
        patches = np.concatenate([np.expand_dims(thumb_patches, 0), patches], 0)

        patch_ordering = np.where(
            patch_ordering >= 0, patch_ordering + self.tokens_per_image, -1
        )
        patch_ordering = np.concatenate(
            [np.arange(0, self.tokens_per_image), patch_ordering], 0
        )

        per_row = np.full((self.token_length_w,), self.patch_token_id, dtype=np.int32)
        if self.use_column_tokens:
            per_row = np.concatenate([per_row, [self.column_token_id]], 0)
        extra_tokens = np.tile(per_row, [self.token_length_h])
        joint = [[self.start_token_id], extra_tokens, [self.end_token_id]] + joint

        image_tokens = np.concatenate(joint, 0).astype(np.int32)

        img_masks = np.pad(img_masks, [[1, 0], [0, 0]], constant_values=1.0)

        return patches, image_tokens, patch_ordering, img_masks

    def process_image(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = np.array(image, dtype=np.float32) / 255.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        new_h, new_w = smart_resize(
            h,
            w,
            factor=self.patch_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if (new_h, new_w) != (h, w):
            image, _ = self.resize_image(image, (new_h, new_w))

        patches, image_tokens, patch_ordering, masks = self.crop_image(image)

        image_input_idx = self.build_image_input_idx(image_tokens, patch_ordering)

        return {
            "pixel_values": patches,
            "image_tokens": image_tokens,
            "image_input_idx": image_input_idx,
            "image_masks": masks,
        }

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        **kwargs,
    ) -> Dict[str, mx.array]:
        if not isinstance(images, list):
            images = [images]

        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)

        results = {
            "pixel_values": [],
            "image_tokens": [],
            "image_input_idx": [],
            "image_masks": [],
        }

        for image in loaded_images:
            processed = self.process_image(image)
            for key in results:
                results[key].append(processed[key])

        return results
