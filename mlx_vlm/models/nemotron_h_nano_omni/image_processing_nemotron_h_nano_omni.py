"""
Image processor for Nemotron-3 Nano Omni.

Port of NVIDIA's reference implementation:
https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/main/image_processing.py
"""

import math
from typing import Optional, Union

import numpy as np
from PIL import Image
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import (
    ImageInput,
    ImageType,
    get_image_type,
    make_list_of_images,
)
from transformers.utils import TensorType


class NemotronHNanoOmniImageProcessor(BaseImageProcessorFast):
    """Dynamic-resolution image processor matching NVIDIA's reference and vLLM's tiler."""

    model_input_names = ["pixel_values"]
    _is_video_mode: bool = False

    def __init__(
        self,
        norm_mean=None,
        norm_std=None,
        patch_size=16,
        downsample_ratio=0.5,
        min_num_patches=1024,
        max_num_patches=13312,
        max_model_len=16384,
        video_target_num_patches=1024,
        video_maintain_aspect_ratio=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self._downsample_factor = int(round(1.0 / downsample_ratio))
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.max_model_len = max_model_len
        self.video_target_num_patches = video_target_num_patches
        self.video_maintain_aspect_ratio = video_maintain_aspect_ratio

    def _process_image(self, image: ImageInput, **kwargs):
        if get_image_type(image) == ImageType.PIL:
            if image.mode != "RGB":
                image = image.convert("RGB")
        return image

    process_image = _process_image

    def _preprocess(
        self,
        images,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        is_video = self._is_video_mode
        images = make_list_of_images(images)

        target_sizes = []
        if is_video:
            for img in images:
                target_w_patches, target_h_patches = self._compute_target_patches_video(
                    img
                )
                target_sizes.append((target_w_patches, target_h_patches))
        else:
            num_tokens_available = self.max_model_len - 4
            budget = num_tokens_available * (self._downsample_factor**2)
            budget = max(budget, self.min_num_patches * len(images))
            max_budget = (
                self.max_num_patches
                if (self.max_num_patches and self.max_num_patches > 0)
                else float("inf")
            )
            per_image_budget = [
                max(min(budget, max_budget), self.min_num_patches) for _ in images
            ]
            for img, tokens_for_media in zip(images, per_image_budget):
                target_w_patches, target_h_patches = self._compute_target_patches(
                    img, tokens_for_media
                )
                target_sizes.append((target_w_patches, target_h_patches))

        norm_mean = np.asarray(self.norm_mean, dtype=np.float32).reshape(3, 1, 1)
        norm_std = np.asarray(self.norm_std, dtype=np.float32).reshape(3, 1, 1)

        pixel_values_list = []
        num_tokens_per_image = []
        imgs_sizes = []
        for img, (wp, hp) in zip(images, target_sizes):
            target_w = wp * self.patch_size
            target_h = hp * self.patch_size
            if img.size != (target_w, target_h):
                # PIL's BICUBIC isn't antialiased like torch's, but with reducing_gap
                # it pre-filters during downsample — close to the reference for
                # typical token-bounded sizes.
                img = img.resize(
                    (target_w, target_h),
                    resample=Image.Resampling.BICUBIC,
                    reducing_gap=3.0,
                )
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)  # (3, H, W)
            arr = (arr / 255.0 - norm_mean) / norm_std
            pixel_values_list.append(arr)
            num_tokens_per_image.append((wp * hp) // (self._downsample_factor**2))
            imgs_sizes.append((target_h, target_w))

        all_same_shape = all(
            t.shape == pixel_values_list[0].shape for t in pixel_values_list
        )
        if all_same_shape:
            pixel_values = np.stack(pixel_values_list, axis=0)
        else:
            pixel_values = pixel_values_list

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "num_patches": [1] * len(images),
                "num_tokens": num_tokens_per_image,
                "imgs_sizes": imgs_sizes,
            },
            tensor_type=(return_tensors if all_same_shape else None),
        )

    def _compute_target_patches(self, img: Image.Image, tokens_available: int):
        orig_w, orig_h = img.width, img.height
        closest_patch_h = round(orig_h / self.patch_size + 0.5)
        closest_patch_w = round(orig_w / self.patch_size + 0.5)
        patches = closest_patch_h * closest_patch_w

        factor = min(math.sqrt(tokens_available / patches), 1.0)
        target_h = math.floor(factor * closest_patch_h)
        target_w = math.floor(factor * closest_patch_w)

        if (
            tokens_available > self.min_num_patches
            and target_h * target_w < self.min_num_patches
        ):
            up = math.sqrt(self.min_num_patches / (target_h * target_w))
            target_h = math.ceil(up * target_h)
            target_w = math.ceil(up * target_w)

        divisor = self._downsample_factor
        rem_h = target_h % divisor
        if rem_h:
            inc_h = divisor - rem_h
            if (target_h + inc_h) * target_w <= tokens_available:
                target_h += inc_h
            else:
                target_h = max(divisor, target_h - rem_h)
        rem_w = target_w % divisor
        if rem_w:
            inc_w = divisor - rem_w
            if target_h * (target_w + inc_w) <= tokens_available:
                target_w += inc_w
            else:
                target_w = max(divisor, target_w - rem_w)

        return target_w, target_h

    def _compute_target_patches_video(self, img: Image.Image):
        orig_w, orig_h = img.width, img.height
        target = self.video_target_num_patches
        divisor = self._downsample_factor
        if self.video_maintain_aspect_ratio:
            aspect_wh = orig_w / max(orig_h, 1)
            ph = max(round(math.sqrt(target / aspect_wh)), 1)
            pw = max(round(math.sqrt(target * aspect_wh)), 1)
            if divisor > 1:
                rem_h = ph % divisor
                rem_w = pw % divisor
                ph_up = ph + (divisor - rem_h if rem_h else 0)
                ph_down = ph - rem_h
                pw_up = pw + (divisor - rem_w if rem_w else 0)
                pw_down = pw - rem_w
                if ph_up * pw_up <= target:
                    ph, pw = ph_up, pw_up
                else:
                    ph = max(divisor, ph_down)
                    pw = max(divisor, pw_down)
        else:
            side = int(math.sqrt(target))
            side = max(divisor, (side // divisor) * divisor)
            ph = pw = side
        return pw, ph


__all__ = ["NemotronHNanoOmniImageProcessor"]
