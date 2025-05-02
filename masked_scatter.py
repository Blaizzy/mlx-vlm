import mlx.core as mx


def mlx_masked_scatter(target: mx.array, mask: mx.array, source: mx.array) -> mx.array:
    """
    Implements masked_scatter functionality with MLX.

    Args:
        target: The tensor to be updated
        mask: Boolean mask indicating which elements to update (same shape as target)
        source: Tensor containing values to copy into target (flattened and used sequentially)

    Returns:
        Updated target tensor

    Raises:
        ValueError: If the mask shape does not match the target shape
        ValueError: If the source tensor has fewer elements than required by the mask
    """
    if mask.shape != target.shape:
        raise ValueError(
            f"Mask shape {mask.shape} must match target shape {target.shape}"
        )

    # Flatten tensors
    target_shape = target.shape
    ndim = len(target_shape)
    mask_flat = mx.flatten(mask)
    source_flat = mx.flatten(source)
    source_len = source_flat.shape[0]

    result = mx.array(target)

    src_idx = 0
    for flat_idx in range(mask_flat.shape[0]):
        if mask_flat[flat_idx].item():
            # Convert flat index to multi-dimensional indices
            indices = []
            temp_idx = flat_idx
            for dim in reversed(target_shape[1:]):
                indices.insert(0, temp_idx % dim)
                temp_idx //= dim
            indices.insert(0, temp_idx)

            # Get the next source value
            if src_idx >= source_len:
                raise ValueError(
                    "Source tensor has fewer elements than required by the mask"
                )
            src_val = source_flat[src_idx]
            src_idx += 1

            # Convert list of indices to array for slice_update
            start_indices = mx.array(indices)

            # Define the axes (0, 1, 2, ..., ndim-1)
            axes = tuple(range(ndim))

            # Create a properly shaped update array
            update_shape = tuple(1 for _ in range(ndim))
            update = mx.reshape(src_val, update_shape)

            # Update the result
            result = mx.slice_update(result, update, start_indices, axes)

    return result


# 2d
target = mx.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]])
source = mx.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
assert mx.allclose(
    mlx_masked_scatter(target, mask, source),
    mx.array([[0, 0, 0, 0, 1], [2, 3, 0, 4, 5]]),
)

# 3d
# fmt: off
target = mx.zeros((1, 5, 3))
mask = mx.array([[[0, 0, 0],[1, 1, 1],[0, 0, 0],[1, 1, 1],[0, 0, 0],]])
source = mx.array([
    [[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12],],
    [[13, 14, 15],[16, 17, 18],[19, 20, 21],[22, 23, 24],],
])
assert mx.allclose(
    mlx_masked_scatter(target, mask, source),
    mx.array([[[0, 0, 0], [1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]]]),
)
# fmt: on
