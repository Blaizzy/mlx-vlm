"""Projection helpers shared by model families."""

import mlx.core as mx
import mlx.nn as nn

DECODE_BLOCK_SIZE = 8


def tokenwise(operation, x, *args, axis=1):
    """Apply an operation with the same reduction geometry at each position."""

    def at(value, index):
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(index, index + 1)
        return mx.contiguous(value[tuple(slices)])

    outputs = [
        operation(at(x, i), *(at(arg, i) for arg in args)) for i in range(x.shape[axis])
    ]
    if isinstance(outputs[0], (tuple, list)):
        return tuple(mx.concatenate(parts, axis=axis) for parts in zip(*outputs))
    return mx.concatenate(outputs, axis=axis)


def native_batch_linear(module, x):
    """Match independent B×1 calls, including dense and mixed-dtype fallbacks."""
    if x.ndim != 3 or x.shape[1] <= 1:
        return module(x)
    # One GEMV dispatch preserves the reduction used by singleton decode.
    # Batched AR may use different reduction geometry, so keep its native path.
    dense = (
        getattr(module, "__self__", module)
        if getattr(module, "__name__", None) == "as_linear"
        else module
    )
    if (
        x.shape[0] == 1
        and x.dtype in (mx.bfloat16, mx.float16)
        and mx.default_device() == mx.gpu
        and isinstance(dense, (nn.Linear, nn.Embedding))
        and not dense.training
        and "bias" not in dense
    ):
        from .exact_speculative_verify import exact_speculative_verify_weight

        result = exact_speculative_verify_weight(dense.weight, x)
        if result is not None:
            return result
    from .quantized_verifier import exact_quantized_linear, singleton_quantized_linear

    # Narrow and FP32 projections can use a different native QMV reduction.
    if (
        x.shape[0] == 1
        and x.shape[-1] % 512 == 0
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        output = singleton_quantized_linear(module, x)
        if output is not None:
            return output
    output = exact_quantized_linear(module, x)
    return tokenwise(module, x) if output is None else output


def linear(module, x):
    """Use decode-equivalent projections for short blocks, GEMM for prefill."""
    if (
        x.ndim == 3
        and 1 < x.shape[1] <= DECODE_BLOCK_SIZE
        and not getattr(module, "training", False)
    ):
        return native_batch_linear(module, x)
    return module(x)


def tiled_linear(module, x):
    """Give large prefill projections a fixed reduction geometry.

    Explicit calls prevent matmul from folding the tile axis into its row
    dimension. The final tile is padded so it uses the same reduction too.
    Small calls retain the model's native geometry, including draft replay.
    """
    tile_size = 256
    if x.ndim != 3 or x.shape[1] <= tile_size:
        return linear(module, x)
    length = x.shape[1]
    padding = (-length) % tile_size
    if padding:
        x = mx.pad(x, [(0, 0), (0, padding), (0, 0)])
    return mx.concatenate(
        [
            module(mx.contiguous(x[:, start : start + tile_size]))
            for start in range(0, length, tile_size)
        ],
        axis=1,
    )[:, :length]
