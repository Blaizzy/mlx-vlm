import mlx.core as mx
import numpy as np


def gaussian_blur_axis(image, sigma, axis):
    """
    Applies a 1D Gaussian blur along the given axis.
    This version works for arrays with any number of dimensions.
    """
    radius = int(3 * sigma)
    if radius < 1:
        return image
    x = mx.arange(-radius, radius + 1)
    kernel = mx.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / mx.sum(kernel)

    # MLX doesn't have a direct apply_along_axis equivalent,
    # so we'll implement the convolution differently based on the axis

    # Helper function to apply 1D convolution along specific axis
    def conv_1d(array, kernel, axis):
        # Reshape kernel to broadcast along the right dimensions
        kernel_shape = [1] * image.ndim
        kernel_shape[axis] = len(kernel)
        kernel_reshaped = kernel.reshape(kernel_shape)

        # Pad the array
        pad_width = [(0, 0)] * image.ndim
        pad_width[axis] = (radius, radius)
        padded = mx.pad(array, pad_width, mode="edge")

        # Perform convolution via sliding window sum
        result = mx.zeros_like(array)
        slices = [slice(None)] * padded.ndim

        for i in range(2 * radius + 1):
            slices[axis] = slice(i, i + array.shape[axis])
            result = result + padded[tuple(slices)] * kernel_reshaped

        return result

    return conv_1d(image, kernel, axis)


def bilinear_interpolate(image, new_height, new_width, align_corners=False):
    """
    Performs bilinear interpolation on an array whose spatial dimensions are the first two.
    It supports extra dimensions (e.g. channels or batch dimensions that have been moved to the trailing axes).
    """
    # image is assumed to have shape (H, W, ...) where H and W are spatial dimensions.
    H_in, W_in = image.shape[0], image.shape[1]

    # Compute sampling positions in the input image.
    if new_height == 1:
        row_positions = mx.array([0.0])
    else:
        if align_corners:
            row_positions = mx.linspace(0, H_in - 1, new_height)
        else:
            row_positions = (mx.arange(new_height) + 0.5) * H_in / new_height - 0.5

    if new_width == 1:
        col_positions = mx.array([0.0])
    else:
        if align_corners:
            col_positions = mx.linspace(0, W_in - 1, new_width)
        else:
            col_positions = (mx.arange(new_width) + 0.5) * W_in / new_width - 0.5

    # Compute floor and ceil indices.
    row_floor = mx.floor(row_positions).astype(mx.int32)
    col_floor = mx.floor(col_positions).astype(mx.int32)
    row_ceil = row_floor + 1
    col_ceil = col_floor + 1

    row_floor = mx.clip(row_floor, 0, H_in - 1)
    row_ceil = mx.clip(row_ceil, 0, H_in - 1)
    col_floor = mx.clip(col_floor, 0, W_in - 1)
    col_ceil = mx.clip(col_ceil, 0, W_in - 1)

    row_weight = row_positions - row_floor  # shape (new_height,)
    col_weight = col_positions - col_floor  # shape (new_width,)

    # Use advanced indexing for gather operations
    # Create meshgrid for coordinates
    row_floor_grid, col_floor_grid = mx.meshgrid(row_floor, col_floor, indexing="ij")
    row_ceil_grid, col_floor_grid = mx.meshgrid(row_ceil, col_floor, indexing="ij")
    row_floor_grid, col_ceil_grid = mx.meshgrid(row_floor, col_ceil, indexing="ij")
    row_ceil_grid, col_ceil_grid = mx.meshgrid(row_ceil, col_ceil, indexing="ij")

    # Gather the four surrounding pixels using take_along_axis
    # For higher dimensional arrays, we'll need to reshape and broadcast
    extra_dims = image.ndim - 2

    def gather_pixels(row_indices, col_indices):
        # Flatten the spatial dimensions for gathering
        flat_indices = row_indices * W_in + col_indices
        flat_image = mx.reshape(image, (-1,) + image.shape[2:])
        # Gather and reshape back
        gathered = mx.take(flat_image, flat_indices.reshape(-1), axis=0)
        return mx.reshape(gathered, (new_height, new_width) + image.shape[2:])

    top_left = gather_pixels(row_floor_grid, col_floor_grid)
    top_right = gather_pixels(row_floor_grid, col_ceil_grid)
    bottom_left = gather_pixels(row_ceil_grid, col_floor_grid)
    bottom_right = gather_pixels(row_ceil_grid, col_ceil_grid)

    # Expand the weights to have shape (new_height, new_width, *[1]*extra_dims)
    r_weight = row_weight.reshape(new_height, 1, *([1] * extra_dims))
    c_weight = col_weight.reshape(1, new_width, *([1] * extra_dims))

    # Perform bilinear interpolation.
    result = (
        (1 - r_weight) * (1 - c_weight) * top_left
        + (1 - r_weight) * c_weight * top_right
        + r_weight * (1 - c_weight) * bottom_left
        + r_weight * c_weight * bottom_right
    )
    return result


def resize_bilinear(image, new_size, align_corners=False, antialias=True):
    """
    Resizes an image (or embedding tensor) to new_size=(new_height, new_width)
    using bilinear interpolation with MLX.

    Supports:
      - 2D: (H, W)
      - 3D: (H, W, C)
      - 4D: (B, C, H, W)  (assumed for typical image batches)
    """
    new_height, new_width = new_size

    # Convert numpy arrays to MLX arrays if needed
    if isinstance(image, np.ndarray):
        image = mx.array(image)

    if image.ndim == 2 or image.ndim == 3:
        # Assume spatial dims are the first two.
        resized = image
        H_in, W_in = image.shape[:2]
        if antialias:
            if new_height < H_in:
                scale_y = new_height / H_in
                sigma_y = (1 / scale_y - 1) / 2.0  # heuristic
                if sigma_y > 0:
                    resized = gaussian_blur_axis(resized, sigma_y, axis=0)
            if new_width < W_in:
                scale_x = new_width / W_in
                sigma_x = (1 / scale_x - 1) / 2.0
                if sigma_x > 0:
                    resized = gaussian_blur_axis(resized, sigma_x, axis=1)
        resized = bilinear_interpolate(
            resized, new_height, new_width, align_corners=align_corners
        )
        return resized

    elif image.ndim == 4:
        # Assume shape is (B, C, H, W) (typical PyTorch/MLX format).
        B, C, H_in, W_in = image.shape
        # Permute to bring spatial dims to the front: (H, W, B, C)
        image_perm = mx.transpose(image, (2, 3, 0, 1))
        resized = image_perm
        if antialias:
            if new_height < H_in:
                scale_y = new_height / H_in
                sigma_y = (1 / scale_y - 1) / 2.0
                if sigma_y > 0:
                    resized = gaussian_blur_axis(resized, sigma_y, axis=0)
            if new_width < W_in:
                scale_x = new_width / W_in
                sigma_x = (1 / scale_x - 1) / 2.0
                if sigma_x > 0:
                    resized = gaussian_blur_axis(resized, sigma_x, axis=1)
        resized = bilinear_interpolate(
            resized, new_height, new_width, align_corners=align_corners
        )
        # Permute back to (B, C, new_height, new_width)
        resized = mx.transpose(resized, (2, 3, 0, 1))
        return resized

    else:
        raise ValueError("Unsupported image dimensions.")


#
