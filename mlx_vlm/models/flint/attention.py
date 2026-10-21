"""Optional Flint prefill paths; decode and quantized caches use the baseline."""

import mlx.core as mx

from ..base import scaled_dot_product_attention
from ..cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache


def unpadded_full_attention_caches(layers, caches):
    """Check mutable padding once per forward, including restored/merged caches."""
    candidates = [
        cache
        for layer, cache in zip(layers, caches)
        if layer.self_attn.window_size is None
        and type(cache) is BatchKVCache
        and getattr(cache, "_right_padding", None) is None
    ]
    if not candidates:
        return set()
    # Do not cache this decision by array identity: padding can change in place
    # during continuous batching or speculative rollback. One combined host
    # read avoids a separate synchronization for every global-attention layer.
    padding = mx.concatenate([cache.left_padding for cache in candidates])
    if mx.all(padding <= 0).item():
        return {id(cache) for cache in candidates}
    return set()


def prefill_attention(q, k, v, *, cache, scale, mask, sinks, window_size, tile_size):
    """Tile queries and their window union while preserving the original mask."""
    length = q.shape[-2]
    offset = k.shape[-2] - length if isinstance(k, mx.array) else -1
    if (
        tile_size <= 0
        or length <= tile_size
        or window_size is None
        or offset < 0
        or mask is None
        or (isinstance(mask, mx.array) and mask.ndim < 2)
        or (isinstance(mask, str) and k.shape[-2] > window_size)
        or type(cache)
        not in (
            type(None),
            KVCache,
            BatchKVCache,
            RotatingKVCache,
            BatchRotatingKVCache,
        )
        or (isinstance(cache, RotatingKVCache) and cache.keep != 0)
    ):
        return scaled_dot_product_attention(
            q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks
        )

    boundaries = list(range(0, length, tile_size)) + [length]
    # A tiny last tile selects a different decode-style SDPA reduction. Keep it
    # with the preceding prefill tile instead of introducing extra BF16 drift.
    if tile_size >= 32 and len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < 32:
        del boundaries[-2]
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        # Retain masked keys up to the preceding 128-key boundary so the
        # reduction blocks stay aligned with the original attention call.
        first_key = max(0, offset + start - window_size + 1) // 128 * 128
        last_key = offset + end
        tile_mask = mask
        if isinstance(mask, mx.array):
            query_slice = slice(None) if mask.shape[-2] == 1 else slice(start, end)
            tile_mask = mask[..., query_slice, first_key:last_key]
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                q[..., start:end, :],
                k[..., first_key:last_key, :],
                v[..., first_key:last_key, :],
                scale=scale,
                mask=tile_mask,
                sinks=sinks,
            )
        )
    return mx.concatenate(outputs, axis=-2)
