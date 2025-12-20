"""Image processor for Molmo2 - MLX-native implementation without torch dependency."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)

# Special tokens
IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_LOW_RES_TOKEN = "<im_low>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
IM_END_TOKEN = "<im_end>"
FRAME_END_TOKEN = "<frame_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"


def normalize_image(
    image: np.ndarray,
    image_mean: List[float],
    image_std: List[float],
) -> np.ndarray:
    """Normalize image with mean and std."""
    image = image.astype(np.float32)
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image_pil(
    image: np.ndarray,
    desired_output_size: List[int],
    resample: PILImageResampling,
) -> np.ndarray:
    """Resize image using PIL instead of torch."""
    # Convert numpy to PIL
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    pil_image = Image.fromarray(image_uint8)

    # Map PILImageResampling to PIL resampling
    resample_map = {
        PILImageResampling.NEAREST: Image.Resampling.NEAREST,
        PILImageResampling.BILINEAR: Image.Resampling.BILINEAR,
        PILImageResampling.BICUBIC: Image.Resampling.BICUBIC,
        PILImageResampling.LANCZOS: Image.Resampling.LANCZOS,
        PILImageResampling.BOX: Image.Resampling.BOX,
        PILImageResampling.HAMMING: Image.Resampling.HAMMING,
    }
    pil_resample = resample_map.get(resample, Image.Resampling.BILINEAR)

    # Resize (PIL uses width, height order)
    resized = pil_image.resize(
        (desired_output_size[1], desired_output_size[0]), resample=pil_resample
    )

    # Convert back to numpy and normalize to [0, 1]
    resized_np = np.array(resized, dtype=np.float32) / 255.0
    return resized_np


def select_tiling(
    h: int, w: int, patch_size: int, max_num_crops: int
) -> Tuple[int, int]:
    """Divide an image of size [w, h] into up to max_num_crops of size patch_size."""
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i * j <= max_num_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([h, w], dtype=np.float32)

    with np.errstate(divide="ignore"):
        required_scale = np.min(
            candidate_resolutions.astype(np.float32) / original_size, axis=-1
        )

    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 1e9, required_scale)
        ix = np.argmin(required_scale)

    return tuple(candidate_tilings[ix])


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: List[int],
    resample: PILImageResampling,
    image_mean: List[float],
    image_std: List[float],
    image_patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a single resized image crop."""
    resized = resize_image_pil(image, base_image_input_size, resample)
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h).reshape(
        [crop_patch_h, crop_patch_w]
    )
    return resized, resize_idx


def build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: List[int],
    base_image_input_size: List[int],
    resample: PILImageResampling,
    image_mean: List[float],
    image_std: List[float],
    image_patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose an image into overlapping crops."""
    crop_size = base_image_input_size[0]
    assert base_image_input_size[0] == base_image_input_size[1]

    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)
    crop_patches = base_image_input_size[0] // image_patch_size
    crop_window_patches = crop_patches - (right_margin + left_margin)
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    original_image_h, original_image_w = image.shape[:2]

    tiling = select_tiling(
        original_image_h - total_margin_pixels,
        original_image_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src = resize_image_pil(
        image,
        [
            tiling[0] * crop_window_size + total_margin_pixels,
            tiling[1] * crop_window_size + total_margin_pixels,
        ],
        resample,
    )
    src = normalize_image(src, image_mean, image_std)

    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)

    on_crop = 0
    for i in range(tiling[0]):
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
            patch_idx = np.arange(crop_patch_w * crop_patch_h).reshape(
                crop_patch_h, crop_patch_w
            )
            patch_idx += on_crop * crop_patch_h * crop_patch_w

            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0] - 1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # Transpose the patch_idx_arr to get the full index array
    patch_idx_full = np.zeros(
        [
            tiling[0] * crop_window_patches + left_margin + right_margin,
            tiling[1] * crop_window_patches + left_margin + right_margin,
        ],
        dtype=np.int32,
    )
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            crop_idx = i * tiling[1] + j
            y_start = i * crop_window_patches
            x_start = j * crop_window_patches
            patch_idx_full[
                y_start : y_start + crop_patch_h, x_start : x_start + crop_patch_w
            ] = np.where(
                patch_idx_arr[crop_idx] >= 0,
                patch_idx_arr[crop_idx],
                patch_idx_full[
                    y_start : y_start + crop_patch_h, x_start : x_start + crop_patch_w
                ],
            )

    return crop_arr, patch_idx_full, tiling


def batch_pixels_to_patches(crops: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert image crops to patches."""
    n_crops, h, w, c = crops.shape
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    n_patches = n_patches_h * n_patches_w
    patch_dim = patch_size * patch_size * c

    # Reshape to patches
    crops = crops.reshape(n_crops, n_patches_h, patch_size, n_patches_w, patch_size, c)
    crops = crops.transpose(0, 1, 3, 2, 4, 5)
    crops = crops.reshape(n_crops, n_patches, patch_dim)
    return crops


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    """Build pooling indices using centered padding (matches HuggingFace implementation)."""
    h, w = idx_arr.shape
    # Calculate padding to make dimensions divisible by pool size (centered padding)
    h_pad = pool_h * ((h + pool_h - 1) // pool_h) - h
    w_pad = pool_w * ((w + pool_w - 1) // pool_w) - w

    # Apply centered padding
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )

    # Rearrange into pooling windows: (h dh) (w dw) -> h w (dh dw)
    padded_h, padded_w = idx_arr.shape
    out_h = padded_h // pool_h
    out_w = padded_w // pool_w

    # Reshape to separate pooling dimensions
    idx_arr = idx_arr.reshape(out_h, pool_h, out_w, pool_w)
    # Transpose to get (out_h, out_w, pool_h, pool_w)
    idx_arr = idx_arr.transpose(0, 2, 1, 3)
    # Reshape to (out_h, out_w, pool_h * pool_w)
    idx_arr = idx_arr.reshape(out_h, out_w, pool_h * pool_w)

    return idx_arr


def image_to_patches_and_grids(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: List[int],
    base_image_input_size: List[int],
    resample: PILImageResampling,
    image_mean: List[float],
    image_std: List[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Convert image to patches with pooling information.

    Returns crops and pooling indices in the order expected by the model:
    - Crops: [low_res_crop, high_res_crops...] (low-res first)
    - Pooling indices: [low_res_pooling, high_res_pooling] (low-res first)
    - Image grid: [lo_h, lo_w, hi_h, hi_w] (low-res dimensions first)
    """
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size

    # Build overlapping crops for high-res
    crop_arr, patch_idx_arr, tiling = build_overlapping_crops(
        image,
        max_crops,
        overlap_margins,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )

    # Build pooling indices for high-res using centered padding (matches HF)
    pooling_idx = arange_for_pooling(patch_idx_arr, image_pooling_h, image_pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape(-1, image_pooling_h * image_pooling_w)

    # Build resized image for low-res
    resize_arr, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )

    # Combine crops: LOW-RES FIRST (matches HuggingFace)
    all_crops = np.concatenate([resize_arr, crop_arr], axis=0)

    # Build pooling indices for low-res
    resize_pooling_idx = arange_for_pooling(
        resize_idx, image_pooling_h, image_pooling_w
    )
    resized_h, resized_w = resize_pooling_idx.shape[:2]
    resize_pooling_idx = resize_pooling_idx.reshape(
        -1, image_pooling_h * image_pooling_w
    )

    # LOW-RES crop is first, so offset HIGH-RES indices by the number of low-res patches
    # (matches HuggingFace: "Global image goes first, so the order of patches in previous crops gets increased")
    pooling_idx = np.where(
        pooling_idx >= 0,
        pooling_idx
        + crop_patch_h * crop_patch_w,  # Offset by one crop (the low-res crop)
        -1,
    )

    # Concatenate pooling indices: LOW-RES FIRST (matches HuggingFace)
    pooling_idx = np.concatenate([resize_pooling_idx, pooling_idx], axis=0)

    # Image grid format: [resized_h, resized_w, h, w] = [lo_h, lo_w, hi_h, hi_w]
    # (matches HuggingFace order)
    image_grid = np.array([[resized_h, resized_w, h, w]], dtype=np.int32)

    return (
        image_grid,
        batch_pixels_to_patches(all_crops, image_patch_size),
        pooling_idx,
        (h, w),  # Return high-res pooled dims for token generation
    )


class Molmo2ImageProcessor(BaseImageProcessor):
    """
    MLX-native image processor for Molmo2 that doesn't require torch.
    """

    model_input_names = [
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
    ]

    def __init__(
        self,
        size: Optional[dict] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        max_crops: int = 8,
        overlap_margins: List[int] = None,
        patch_size: int = 14,
        pooling_size: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 378, "width": 378}
        size = get_size_dict(size, default_to_square=True)
        self.size = size

        self.resample = resample
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

        self.max_crops = max_crops
        self.overlap_margins = (
            overlap_margins if overlap_margins is not None else [4, 4]
        )
        self.patch_size = patch_size
        self.pooling_size = pooling_size if pooling_size is not None else [2, 2]

    def preprocess(
        self,
        images: ImageInput,
        size: Optional[dict] = None,
        resample: Optional[PILImageResampling] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: Optional[bool] = None,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        patch_size: Optional[int] = None,
        pooling_size: Optional[List[int]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess images for Molmo2."""
        if size is not None:
            if "height" not in size or "width" not in size:
                raise ValueError("size must contain 'height' and 'width' keys.")
        else:
            size = {**self.size}

        base_image_input_size = [size["height"], size["width"]]

        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        if images is not None:
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        data = {}
        if images is not None:
            batch_grids = []
            batch_crops = []
            batch_pooled_patches_idx = []
            batch_num_crops = []
            batch_hi_pooled_dims = []

            for image in images:
                image_grid, crops, pooled_idx, hi_pooled_dims = (
                    image_to_patches_and_grids(
                        image,
                        max_crops,
                        overlap_margins,
                        base_image_input_size,
                        resample,
                        image_mean,
                        image_std,
                        patch_size,
                        image_pooling_w,
                        image_pooling_h,
                    )
                )
                batch_grids.append(image_grid)
                batch_crops.append(crops)
                batch_pooled_patches_idx.append(pooled_idx)
                batch_num_crops.append(crops.shape[0])
                batch_hi_pooled_dims.append(hi_pooled_dims)

            pixel_values = np.concatenate(batch_crops, 0)
            image_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
            image_grids = np.concatenate(batch_grids, 0)
            image_num_crops = np.array(batch_num_crops)

            data.update(
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                _hi_pooled_dims=batch_hi_pooled_dims,  # Internal use for token generation
            )

        return BatchFeature(data, tensor_type=return_tensors)


class Molmo2Processor(ProcessorMixin):
    """
    Processor for Molmo2 that combines image processor and tokenizer.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Molmo2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_use_col_tokens: bool = True,
        use_single_crop_col_tokens: Optional[bool] = None,
        use_single_crop_start_token: bool = True,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = Molmo2ImageProcessor()
        super().__init__(image_processor, tokenizer, **kwargs)
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.image_placeholder_token = IMAGE_PROMPT

    def get_image_tokens(self, image_grid: np.ndarray) -> str:
        """Generate image token string from image grid.

        Args:
            image_grid: Array of [resized_h, resized_w, height, width]
                        = [lo_pooled_h, lo_pooled_w, hi_pooled_h, hi_pooled_w]
                        (matches HuggingFace format)

        Returns:
            String of image tokens to insert into prompt.
        """
        # Unpack in HuggingFace order: [lo_h, lo_w, hi_h, hi_w]
        resized_h, resized_w, height, width = image_grid

        # Build high-res tokens first (will be appended after low-res)
        per_row = [IMAGE_PATCH_TOKEN] * width
        if self.image_use_col_tokens:
            per_row = per_row + [IM_COL_TOKEN]
        hi_res_tokens = [IM_START_TOKEN] + per_row * height + [IM_END_TOKEN]

        # Build low-res tokens
        per_row = [IMAGE_PATCH_TOKEN] * resized_w
        use_single_crop_col_tokens = (
            self.image_use_col_tokens
            if self.use_single_crop_col_tokens is None
            else self.use_single_crop_col_tokens
        )
        image_start_token = (
            LOW_RES_IMAGE_START_TOKEN
            if self.use_single_crop_start_token
            else IM_START_TOKEN
        )
        if use_single_crop_col_tokens:
            per_row = per_row + [IM_COL_TOKEN]
        lo_res_tokens = [image_start_token] + per_row * resized_h + [IM_END_TOKEN]

        # Low-res comes first, then high-res (matches HuggingFace)
        all_tokens = lo_res_tokens + hi_res_tokens
        return "".join(all_tokens)

    def __call__(
        self,
        text=None,
        images=None,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        """Process text and images for the model."""
        encoding = {}
        image_grids = None

        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=None)
            image_grids = image_inputs.get("image_grids")
            # Remove internal keys before adding to encoding
            hi_pooled_dims = image_inputs.pop("_hi_pooled_dims", None)
            encoding.update(image_inputs)

        if text is not None:
            # Expand image placeholders with actual image tokens
            if image_grids is not None:
                if isinstance(text, str):
                    text = [text]
                    was_string = True
                else:
                    text = list(text)
                    was_string = False

                image_idx = 0
                for i in range(len(text)):
                    num_images = text[i].count(self.image_placeholder_token)
                    for _ in range(num_images):
                        if image_idx < len(image_grids):
                            image_tokens = self.get_image_tokens(image_grids[image_idx])
                            text[i] = text[i].replace(
                                self.image_placeholder_token, image_tokens, 1
                            )
                            image_idx += 1

                if was_string:
                    text = text[0]

            text_inputs = self.tokenizer(
                text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
            encoding.update(text_inputs)

        # Convert to requested tensor type
        if return_tensors is not None:
            encoding = BatchFeature(encoding, tensor_type=return_tensors)

        return encoding

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


# Register the custom processor with transformers AutoProcessor
MODEL_TYPE = "molmo2"
try:
    AutoImageProcessor.register(
        MODEL_TYPE, slow_image_processor_class=Molmo2ImageProcessor
    )
    AutoProcessor.register(MODEL_TYPE, Molmo2Processor)
    logger.info(f"Registered custom processor classes for model type '{MODEL_TYPE}'.")
except Exception as e:
    logger.warning(f"Failed to register custom processor for {MODEL_TYPE}: {e}")
