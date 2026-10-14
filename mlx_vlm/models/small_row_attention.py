"""Group qualified query blocks with singleton-equivalent attention reductions."""

from functools import lru_cache
from importlib.metadata import version

import mlx.core as mx


@lru_cache(maxsize=1)
def _supports_invariant_grouping():
    return (
        mx.metal.is_available()
        and mx.device_info().get("architecture") == "applegpu_g15d"
        and version("mlx") == "0.32.3"
    )


def small_row_attention(queries, keys, values, *, scale, mask=None, sinks=None):
    rows = queries.shape[2]
    if rows < 2 or keys.shape[2] < rows:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )

    chunk = 4 if rows > 5 else rows
    if queries.shape[1] == 16 and keys.shape[1] == 2:
        # Five queries with eight heads per KV head change native reductions.
        chunk = min(chunk, 4)
    if not (
        _supports_invariant_grouping()
        and mx.default_device() == mx.gpu
        and queries.shape[0] == 1
        and (
            queries.shape[1] == 6 * keys.shape[1]
            # MiniCPM layouts are separately qualified only below this boundary.
            # Longer contexts use singleton attention; widening this changes bits.
            or queries.shape[1] in (8, 16)
            and keys.shape[1] == 2
            and keys.shape[2] < 16384
            and rows <= 8
        )
        and queries.shape[-1] == values.shape[-1] == 256
        and queries.dtype == keys.dtype == values.dtype == mx.bfloat16
        and sinks is None
        and (mask is None or isinstance(mask, str) and mask == "causal")
    ):
        chunk = 1
    prefix = keys.shape[2] - rows
    causal = isinstance(mask, str) and mask == "causal"
    outputs = []
    start = 0
    while start < rows:
        stop = min(start + chunk, rows)
        if causal:
            # Keep every query in a group on the same native reduction plan
            # as its standalone call. Context length changes that plan here.
            for boundary in (1024, 16384, 65536):
                cut = boundary - 1 - prefix
                if start < cut < stop:
                    stop = cut
        # A causal chunk ends at its last query's key position. This preserves
        # the bottom-right alignment of the native causal mask.
        end = prefix + stop if causal else keys.shape[2]
        chunk_mask = mask
        if isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == rows:
            chunk_mask = mask[..., start:stop, :end]
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, start:stop],
                keys[:, :, :end],
                values[:, :, :end],
                scale=scale,
                mask=chunk_mask,
                sinks=sinks,
            )
        )
        start = stop
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
