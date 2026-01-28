"""Image processor for ERNIE 4.5 VL MoE."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from transformers.image_transforms import (
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

from ..base import BaseImageProcessor


def round_by_factor(number: int, factor: int) -> int:
    """Round number to nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Round up number to nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Round down number to nearest multiple of factor."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> Tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels within ['min_pixels', 'max_pixels']
    3. Aspect ratio maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        factor: Factor to make dimensions divisible by (patch_size * merge_size)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (resized_height, resized_width)
    """
    # Clamp extreme aspect ratios
    MAX_RATIO = 200
    if height / width > MAX_RATIO:
        width = height // MAX_RATIO
    elif width / height > MAX_RATIO:
        height = width // MAX_RATIO

    # Round to nearest factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # Scale down if exceeding max_pixels
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    # Scale up if below min_pixels
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    # Ensure minimum size
    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)

    return h_bar, w_bar


class ImageProcessor(BaseImageProcessor):
    """
    Image processor for ERNIE 4.5 VL MoE model.

    Handles variable resolution images by:
    1. Smart resizing to dimensions divisible by (patch_size * merge_size)
    2. Extracting patches in the format expected by the vision encoder
    3. Computing grid_thw (temporal, height, width in patches)
    """

    def __init__(
        self,
        image_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        image_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        size: Tuple[int, int] = (224, 224),
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        config=None,
        **kwargs,
    ):
        # Extract values from config if provided (can be dict or object)
        if config is not None:
            # Handle both dict (from load_image_processor) and object configs
            if isinstance(config, dict):
                # Get vision_config from the main config dict
                vision_config = config.get("vision_config", {})

                # Extract image processing params from root or vision_config
                image_mean = config.get("image_mean", image_mean)
                image_std = config.get("image_std", image_std)
                min_pixels = config.get("min_pixels", min_pixels)
                max_pixels = config.get("max_pixels", max_pixels)

                # Extract vision params
                patch_size = vision_config.get(
                    "patch_size", config.get("patch_size", patch_size)
                )
                merge_size = vision_config.get(
                    "spatial_merge_size", config.get("spatial_merge_size", merge_size)
                )
                temporal_patch_size = vision_config.get(
                    "temporal_patch_size",
                    config.get("temporal_patch_size", temporal_patch_size),
                )
            else:
                # Object config (VisionConfig or similar)
                patch_size = getattr(config, "patch_size", patch_size)
                merge_size = getattr(
                    config,
                    "spatial_merge_size",
                    getattr(config, "merge_size", merge_size),
                )
                temporal_patch_size = getattr(
                    config, "temporal_patch_size", temporal_patch_size
                )

        super().__init__(
            image_mean=image_mean,
            image_std=image_std,
            size=size,
            resample=resample,
            rescale_factor=rescale_factor,
            data_format=data_format,
        )
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.factor = patch_size * merge_size

    def get_smart_resize(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Compute smart resize dimensions and grid dimensions.

        Args:
            height: Original image height
            width: Original image width
            min_pixels: Override minimum pixels
            max_pixels: Override maximum pixels

        Returns:
            Tuple of ((resized_h, resized_w), (grid_h, grid_w))
        """
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )

        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        return (resized_height, resized_width), (grid_h, grid_w)

    def _extract_patches(
        self,
        image: np.ndarray,
        grid_h: int,
        grid_w: int,
    ) -> np.ndarray:
        """
        Extract patches from image in the format expected by the vision encoder.

        Args:
            image: Image array of shape [C, H, W]
            grid_h: Number of patches in height
            grid_w: Number of patches in width

        Returns:
            Patches of shape [grid_h * grid_w, C * patch_size * patch_size]
        """
        C, H, W = image.shape

        # Reshape to patches with merge_size aggregation
        # [C, H, W] -> [C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
        patches = image.reshape(
            C,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Transpose to group spatial patches together
        # -> [grid_h/merge, grid_w/merge, merge, merge, C, patch, patch]
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)

        # Flatten to [num_patches, C * patch_size * patch_size]
        num_patches = (
            (grid_h // self.merge_size)
            * (grid_w // self.merge_size)
            * (self.merge_size**2)
        )
        patches = patches.reshape(num_patches, C * self.patch_size * self.patch_size)

        return patches

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_grid_thw: bool = True,
    ) -> Union[np.ndarray, Dict]:
        """
        Preprocess images for ERNIE 4.5 VL.

        Args:
            images: Single image or list of images
            return_grid_thw: If True, return dict with pixel_values and image_grid_thw

        Returns:
            If return_grid_thw is True: Dict with 'pixel_values' and 'image_grid_thw'
            Otherwise: numpy array of processed images
        """
        if isinstance(images, Image.Image):
            images = [images]

        all_patches = []
        all_grid_thw = []

        for image in images:
            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get smart resize dimensions
            (resized_h, resized_w), (grid_h, grid_w) = self.get_smart_resize(
                image.height, image.width
            )

            # Convert to numpy
            img_array = to_numpy_array(image)

            # Resize
            img_array = resize(
                img_array,
                size=(resized_h, resized_w),
                resample=self.resample,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Rescale
            img_array = rescale(
                img_array,
                scale=self.rescale_factor,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Normalize
            img_array = normalize(
                img_array,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Convert to channel first [H, W, C] -> [C, H, W]
            img_array = to_channel_dimension_format(
                img_array,
                channel_dim=ChannelDimension.FIRST,
                input_channel_dim=ChannelDimension.LAST,
            )

            # Extract patches
            patches = self._extract_patches(img_array, grid_h, grid_w)
            all_patches.append(patches)

            # Store grid_thw (temporal=1 for images)
            all_grid_thw.append([1, grid_h, grid_w])

        # Concatenate all patches
        pixel_values = np.concatenate(all_patches, axis=0)

        if return_grid_thw:
            return {
                "pixel_values": pixel_values,
                "image_grid_thw": np.array(all_grid_thw, dtype=np.int64),
            }

        return pixel_values

    def preprocess_video(
        self,
        frames: List[Image.Image],
        return_grid_thw: bool = True,
    ) -> Union[np.ndarray, Dict]:
        """
        Preprocess video frames for ERNIE 4.5 VL.

        Args:
            frames: List of video frames as PIL Images
            return_grid_thw: If True, return dict with pixel_values and video_grid_thw

        Returns:
            If return_grid_thw is True: Dict with 'pixel_values' and 'video_grid_thw'
            Otherwise: numpy array of processed frames
        """
        if not frames:
            raise ValueError("frames list cannot be empty")

        # Get dimensions from first frame
        first_frame = frames[0]
        if first_frame.mode != "RGB":
            first_frame = first_frame.convert("RGB")

        (resized_h, resized_w), (grid_h, grid_w) = self.get_smart_resize(
            first_frame.height, first_frame.width
        )

        all_patches = []

        for frame in frames:
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            # Convert to numpy
            img_array = to_numpy_array(frame)

            # Resize
            img_array = resize(
                img_array,
                size=(resized_h, resized_w),
                resample=self.resample,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Rescale
            img_array = rescale(
                img_array,
                scale=self.rescale_factor,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Normalize
            img_array = normalize(
                img_array,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.LAST,
                input_data_format=ChannelDimension.LAST,
            )

            # Convert to channel first
            img_array = to_channel_dimension_format(
                img_array,
                channel_dim=ChannelDimension.FIRST,
                input_channel_dim=ChannelDimension.LAST,
            )

            # Extract patches
            patches = self._extract_patches(img_array, grid_h, grid_w)
            all_patches.append(patches)

        # Stack all frame patches
        pixel_values = np.concatenate(all_patches, axis=0)

        # Compute temporal grid
        num_frames = len(frames)
        grid_t = num_frames

        if return_grid_thw:
            return {
                "pixel_values": pixel_values,
                "video_grid_thw": np.array([[grid_t, grid_h, grid_w]], dtype=np.int64),
            }

        return pixel_values
