# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniCPM-OCR image slicing and adaptive upsize, using Pillow and NumPy.

Adapted from OpenBMB's Apache-2.0 MiniCPM-OCR image processor.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch
from ..minicpmv4_6.processing_minicpmv4_6 import (
    MiniCPMVImageProcessor,
    MiniCPMVProcessor,
)
from ..minicpmv4_7.processing_minicpmv4_7 import MiniCPMV4_7Processor

ADAPTIVE_DEFAULTS = {
    "disable_small_image_upsize": False,
    "adaptive_small_image_upsize": False,
    "adaptive_upsize_font_px": 12.0,
    "adaptive_upsize_dense_per_kpx": 2.0,
    "adaptive_upsize_tiny_pixels": 6272,
    "adaptive_upsize_tiny_target_resolution": 84,
    "adaptive_tile_upsize": False,
}


def adaptive_upsize_tier(
    image, font_px_threshold, dense_per_kpx_threshold, tiny_pixels
):
    import cv2

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binary == 0
    if foreground.mean() > 0.5:
        foreground = ~foreground
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    heights = [
        h
        for _, _, w, h, area in stats[1:count]
        if area >= 3 and h >= 2 and w < 0.9 * width and h < 0.9 * height
    ]
    font_px = float(np.median(heights)) if heights else 0.0
    density = len(heights) * 1000.0 / (width * height)
    if 0 < font_px <= font_px_threshold and density >= dense_per_kpx_threshold:
        return "full"
    return "tiny" if width * height < tiny_pixels else "native"


class MiniCPMOcrImageProcessor(MiniCPMVImageProcessor):
    def __init__(self, downsample_mode="4x", **kwargs):
        for key, default in ADAPTIVE_DEFAULTS.items():
            setattr(self, key, kwargs.pop(key, default))
        super().__init__(downsample_mode=downsample_mode, **kwargs)

    def _find_best_resize(
        self, image_size, scale_resolution, patch_size, allow_upscale=False
    ):
        width, height = image_size
        if width * height > scale_resolution**2 or allow_upscale:
            ratio = width / height
            height = int(scale_resolution / math.sqrt(ratio))
            width = int(height * ratio)
        alignment = patch_size * (2 if self.downsample_mode == "4x" else 4)
        return self._ensure_divide(width, alignment), self._ensure_divide(
            height, alignment
        )

    def _tier(self, image):
        return adaptive_upsize_tier(
            image,
            self.adaptive_upsize_font_px,
            self.adaptive_upsize_dense_per_kpx,
            self.adaptive_upsize_tiny_pixels,
        )

    def _source_size(self, image):
        resolution = self.scale_resolution
        upscale = not self.disable_small_image_upsize
        if self.adaptive_small_image_upsize:
            tier = self._tier(image)
            if tier == "full":
                upscale = True
            elif tier == "tiny":
                resolution, upscale = self.adaptive_upsize_tiny_target_resolution, True
        return self._find_best_resize(image.size, resolution, self.patch_size, upscale)

    def slice_image(self, image, max_slice_nums=None, never_split=False):
        grid = self.get_sliced_grid(
            image.size, max_slice_nums, never_split or not self.slice_mode
        )
        if grid is None:
            return (
                image.resize(self._source_size(image), Image.Resampling.BICUBIC),
                [],
                None,
            )
        source, patches, grid = super().slice_image(image, max_slice_nums, never_split)
        if self.adaptive_tile_upsize and self.adaptive_small_image_upsize:
            width, height = image.size
            cols, rows = grid
            for row in range(rows):
                for col in range(cols):
                    cell = image.crop(
                        (
                            round(width * col / cols),
                            round(height * row / rows),
                            round(width * (col + 1) / cols),
                            round(height * (row + 1) / rows),
                        )
                    )
                    if (
                        cell.width * cell.height > self.scale_resolution**2
                        or min(cell.size) < 1
                    ):
                        continue
                    tier = self._tier(cell)
                    if tier == "full":
                        continue
                    size = self._find_best_resize(
                        cell.size,
                        (
                            self.adaptive_upsize_tiny_target_resolution
                            if tier == "tiny"
                            else self.scale_resolution
                        ),
                        self.patch_size,
                        allow_upscale=tier == "tiny",
                    )
                    baseline = patches[row][col]
                    if size[0] * size[1] < baseline.width * baseline.height:
                        patches[row][col] = cell.resize(size, Image.Resampling.BICUBIC)
        return source, patches, grid

    def preprocess(self, images):
        result = {
            key: []
            for key in (
                "pixel_values",
                "tgt_sizes",
                "image_sizes",
                "grids",
                "num_patches_per_image",
            )
        }
        for sample in images:
            pixels, targets, sizes, grids, counts = [], [], [], [], []
            for image in sample:
                image = self._to_pil_image(image)
                source, rows, grid = self.slice_image(image)
                crops = [source, *(crop for row in rows for crop in row)]
                sizes.append(image.size)
                grids.append(grid or (0, 0))
                counts.append(len(crops))
                for crop in crops:
                    values, target, _ = self._preprocess_resized(crop)
                    pixels.append(values)
                    targets.append(target)
            for key, value in zip(
                result,
                (
                    pixels,
                    np.asarray(targets, dtype=np.int32).reshape(-1, 2),
                    sizes,
                    grids,
                    counts,
                ),
            ):
                result[key].append(value)
        return result


class MiniCPMOcrProcessor(MiniCPMV4_7Processor):
    model_type = "minicpm_ocr"

    def __init__(self, image_processor=None, **kwargs):
        if image_processor is None:
            image_processor = MiniCPMOcrImageProcessor()
        elif not isinstance(image_processor, MiniCPMOcrImageProcessor):
            # Reuse the common MiniCPM tokenizer/config loader, replacing only
            # the OCR-specific image geometry and adaptive upsize policy.
            config = image_processor.to_dict()
            config["norm_mean"] = image_processor.mean.reshape(-1).tolist()
            config["norm_std"] = image_processor.std.reshape(-1).tolist()
            image_processor = MiniCPMOcrImageProcessor(**config)
        super().__init__(image_processor=image_processor, **kwargs)

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        downsample_mode=None,
        slice_mode=None,
        max_slice_nums=None,
        use_image_id=None,
        pack_image_by_patch=None,
        max_num_frames=None,
        stack_frames=None,
        disable_small_image_upsize=None,
        adaptive_small_image_upsize=None,
        adaptive_upsize_font_px=None,
        adaptive_upsize_dense_per_kpx=None,
        adaptive_upsize_tiny_pixels=None,
        adaptive_upsize_tiny_target_resolution=None,
        adaptive_tile_upsize=None,
        padding=True,
        padding_side="left",
        max_length=None,
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        # prepare_inputs forwards processor options by their declared names.
        # Keep the image controls explicit so generation honors them too.
        options = {
            "disable_small_image_upsize": disable_small_image_upsize,
            "adaptive_small_image_upsize": adaptive_small_image_upsize,
            "adaptive_upsize_font_px": adaptive_upsize_font_px,
            "adaptive_upsize_dense_per_kpx": adaptive_upsize_dense_per_kpx,
            "adaptive_upsize_tiny_pixels": adaptive_upsize_tiny_pixels,
            "adaptive_upsize_tiny_target_resolution": adaptive_upsize_tiny_target_resolution,
            "adaptive_tile_upsize": adaptive_tile_upsize,
        }
        options = {key: value for key, value in options.items() if value is not None}
        old = {key: getattr(self.image_processor, key) for key in options}
        for key, value in options.items():
            setattr(self.image_processor, key, value)
        try:
            return super().__call__(
                text=text,
                images=images,
                videos=videos,
                downsample_mode=(
                    self.image_processor.downsample_mode
                    if downsample_mode is None
                    else downsample_mode
                ),
                slice_mode=slice_mode,
                max_slice_nums=max_slice_nums,
                use_image_id=use_image_id,
                pack_image_by_patch=pack_image_by_patch,
                max_num_frames=max_num_frames,
                stack_frames=stack_frames,
                padding=padding,
                padding_side=padding_side,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs,
            )
        finally:
            for key, value in old.items():
                setattr(self.image_processor, key, value)

    def _build_placeholder_ids_for_image(
        self,
        patch_target_sizes,
        grid,
        image_idx,
        token_divisor,
        include_image_id=True,
        slice_mode=None,
        fill_attr="image_token_id",
        fill_token=None,
    ):
        if len(patch_target_sizes) == 0:
            return []
        counts = np.prod(patch_target_sizes, axis=-1) // token_divisor
        ids = self._build_image_id_ids(image_idx) if include_image_id else []
        ids += self._build_image_placeholder_ids(
            int(counts[0]), fill_attr=fill_attr, fill_token=fill_token
        )
        cols, rows = grid
        if slice_mode is None:
            slice_mode = self.image_processor.slice_mode
        if slice_mode and cols > 0 and rows > 0:
            if len(counts) != 1 + rows * cols:
                raise ValueError("Image slice grid does not match crop count.")
            for row in range(rows):
                for col in range(cols):
                    ids += self._build_slice_placeholder_ids(
                        int(counts[1 + row * cols + col]),
                        fill_attr=fill_attr,
                        fill_token=fill_token,
                    )
                if row + 1 < rows:
                    ids += self.tokenizer.encode("\n", add_special_tokens=False)
        return ids

    # OCR, unlike V 4.7, also prefixes videos with a local image ID.
    _build_placeholder_ids_for_video = (
        MiniCPMVProcessor._build_placeholder_ids_for_video
    )

    def save_pretrained(self, save_directory, **kwargs):
        files = super().save_pretrained(save_directory, **kwargs)
        path = Path(save_directory) / "processor_config.json"
        config = json.loads(path.read_text())
        config["image_processor"].update(
            {key: getattr(self.image_processor, key) for key in ADAPTIVE_DEFAULTS}
        )
        path.write_text(json.dumps(config, indent=2) + "\n")
        return files


install_auto_processor_patch("minicpm_ocr", MiniCPMOcrProcessor)
