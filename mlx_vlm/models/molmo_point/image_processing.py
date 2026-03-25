"""Image processor for MolmoPoint - no torch dependency."""

import numpy as np
from PIL import Image

from ..interpolate import resize_bilinear

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]


def normalize_image(image, image_mean, image_std):
    image = image.astype(np.float32)
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(image, desired_output_size):
    """Resize image using the shared bilinear interpolation (no torch dependency)."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.array(image)

    h, w = desired_output_size
    is_uint8 = arr.dtype == np.uint8

    if is_uint8:
        arr = arr.astype(np.float32) / 255.0

    # Use the shared bilinear interpolation (HWC format, no antialias)
    resized = resize_bilinear(
        arr, (int(h), int(w)), align_corners=False, antialias=False
    )
    resized = np.array(resized)

    if is_uint8:
        resized = np.clip(resized, 0.0, 1.0)

    return resized.astype(np.float32)


def select_tiling(h, w, patch_size, max_num_crops):
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
        required_scale_d = (candidate_resolutions.astype(np.float32) / original_size,)
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)
    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def build_resized_image(
    image, base_image_input_size, image_mean, image_std, image_patch_size
):
    resized = resize_image(image, base_image_input_size)
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
    image,
    max_crops,
    overlap_margins,
    base_image_input_size,
    image_mean,
    image_std,
    image_patch_size,
):
    crop_size = base_image_input_size[0]
    assert base_image_input_size[0] == base_image_input_size[1]

    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)
    crop_patches = base_image_input_size[0] // image_patch_size
    crop_window_patches = crop_patches - (right_margin + left_margin)
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size

    tiling = select_tiling(
        image.shape[0] - total_margin_pixels,
        image.shape[1] - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src = resize_image(
        image,
        [
            tiling[0] * crop_window_size + total_margin_pixels,
            tiling[1] * crop_window_size + total_margin_pixels,
        ],
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

    patch_idx_arr = np.reshape(
        patch_idx_arr, [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
    )
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
    patch_idx_arr = np.reshape(patch_idx_arr, [-1])
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0] // image_patch_size,
        src.shape[1] // image_patch_size,
    )
    return crop_arr, patch_idx_arr


def batch_pixels_to_patches(array, patch_size):
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(
            array, [n_crops, h_patches, patch_size, w_patches, patch_size]
        )
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(
            array, [n_crops, h_patches * w_patches, patch_size * patch_size]
        )
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(
            array, [n_crops, h_patches, patch_size, w_patches, patch_size, c]
        )
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(
            array, [n_crops, h_patches * w_patches, patch_size * patch_size * c]
        )
        return array


def arange_for_pooling(idx_arr, pool_h, pool_w):
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    # Replace einops.rearrange: "(h dh) (w dw) -> h w (dh dw)"
    H, W = idx_arr.shape
    h = H // pool_h
    w = W // pool_w
    idx_arr = idx_arr.reshape(h, pool_h, w, pool_w)
    idx_arr = idx_arr.transpose(0, 2, 1, 3)
    idx_arr = idx_arr.reshape(h, w, pool_h * pool_w)
    return idx_arr


def image_to_patches_and_grids(
    image,
    max_crops,
    overlap_margins,
    base_image_input_size,
    image_mean,
    image_std,
    image_patch_size,
    image_pooling_w,
    image_pooling_h,
):
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)

    pooling_w = image_pooling_w
    pooling_h = image_pooling_h
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size

    crop_arr, patch_idx_arr = build_overlapping_crops(
        image,
        max_crops,
        overlap_margins,
        base_image_input_size,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])

    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        image_mean,
        image_std,
        image_patch_size,
    )
    patch_idx_arr += crop_patch_h * crop_patch_w
    crop_arr = np.concatenate([resized, crop_arr], 0)

    resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])

    pooling_idx = np.where(
        pooling_idx >= 0,
        pooling_idx + crop_patch_h * crop_patch_w,
        -1,
    )
    pooling_idx = np.concatenate([resize_idx, pooling_idx])
    image_grid = [np.array([resized_h, resized_w, h, w])]

    return (
        np.stack(image_grid, 0),
        batch_pixels_to_patches(crop_arr, image_patch_size),
        pooling_idx,
        patch_idx_arr,
    )


def preprocess_images(
    images,
    max_crops=8,
    overlap_margins=(4, 4),
    base_image_input_size=(378, 378),
    image_mean=None,
    image_std=None,
    image_patch_size=14,
    pooling_size=(2, 2),
    return_pointing_metadata=False,
):
    """Process a list of PIL images for MolmoPoint."""
    if image_mean is None:
        image_mean = SIGLIP_MEAN
    if image_std is None:
        image_std = SIGLIP_STD

    image_pooling_h, image_pooling_w = pooling_size

    batch_grids = []
    batch_crops = []
    batch_pooled_patches_idx = []
    batch_num_crops = []
    patch_mappings = []
    absolute_token_pooling = []
    offset = 0

    # Convert to numpy
    np_images = []
    for img in images:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            np_images.append(np.array(img))
        elif isinstance(img, np.ndarray):
            np_images.append(img)
        else:
            np_images.append(np.array(img))

    for image in np_images:
        image_grid, crops, pooled_idx, patch_mapping = image_to_patches_and_grids(
            image,
            max_crops,
            overlap_margins,
            base_image_input_size,
            image_mean,
            image_std,
            image_patch_size,
            image_pooling_w,
            image_pooling_h,
        )
        batch_grids.append(image_grid)
        batch_crops.append(crops)
        batch_pooled_patches_idx.append(pooled_idx)
        batch_num_crops.append(crops.shape[0])
        if return_pointing_metadata:
            absolute_token_pooling.append(
                np.where(pooled_idx >= 0, pooled_idx + offset, -1)
            )
            patch_mappings.append(patch_mapping + offset)
            n_patches = np.prod(crops.shape[:2])
            offset += n_patches

    pixel_values = np.concatenate(batch_crops, 0)
    image_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
    image_grids = np.concatenate(batch_grids, 0)
    image_num_crops = np.array(batch_num_crops)

    result = {
        "pixel_values": pixel_values,
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
    }

    if return_pointing_metadata:
        result["_pointing_metadata"] = {
            "token_pooling": (
                np.concatenate(absolute_token_pooling, 0)
                if absolute_token_pooling
                else None
            ),
            "subpatch_mapping": patch_mappings,
            "image_sizes": [(img.shape[1], img.shape[0]) for img in np_images],
        }

    return result
