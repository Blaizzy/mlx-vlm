"""Fused padding, sliding-window, and ring-buffer masks for Flint batches."""

from functools import lru_cache

import mlx.core as mx

from ..base import create_attention_mask as reference_attention_mask
from ..cache import BatchKVCache, BatchRotatingKVCache


@lru_cache(None)
def _mask_kernel():
    return mx.fast.metal_kernel(
        name="flint_batched_attention_mask",
        input_names=["padding", "parameters"],
        output_names=["mask"],
        source="""
        uint i = thread_position_in_grid.x;
        uint n = parameters[0], k = parameters[1], batch = parameters[2];
        if (i >= batch * n * k) return;
        uint row = i / (n * k), query = (i / k) % n, column = i % k;
        uint key = (column + k - parameters[5]) % k;
        int left = int(padding[row]) - int(parameters[6]);
        uint position = query + parameters[3];
        mask[i] = position >= key &&
            (parameters[4] == 0 || position < key + parameters[4]) &&
            int(key) >= left;
        """,
    )


def create_attention_mask(
    x, cache=None, window_size=None, return_array=False, *, parameter_cache=None
):
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or type(cache) not in (BatchKVCache, BatchRotatingKVCache)
        or x.shape[1] == 0
    ):
        return reference_attention_mask(x, cache, window_size, return_array)

    length = x.shape[1]
    shift = trim = 0
    if type(cache) is BatchRotatingKVCache:
        window_size = window_size or cache.max_size
        offset = min(cache.max_size - 1, cache._offset)
        in_place = length == 1 and cache._lengths is None
        trim = max(0, cache._idx - cache.max_size + int(not in_place))
        if in_place and (cache.rotated or cache._idx >= cache.max_size):
            trim += 1
            shift = (0 if cache._idx >= cache.max_size else cache._idx) + 1
    else:
        offset = cache._idx

    keys = offset + length
    batch = cache.left_padding.size
    geometry = (length, keys, batch, offset, window_size or 0, shift % keys, trim)
    # Share only scalar geometry within this forward. Padding stays an input
    # from each individual cache, even when layer positions happen to match.
    parameters = None if parameter_cache is None else parameter_cache.get(geometry)
    if parameters is None:
        parameters = mx.array(geometry, mx.uint32)
        if parameter_cache is not None:
            parameter_cache[geometry] = parameters
    return _mask_kernel()(
        inputs=[cache.left_padding, parameters],
        grid=(batch * length * keys, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, 1, length, keys)],
        output_dtypes=[mx.bool_],
    )[0]
