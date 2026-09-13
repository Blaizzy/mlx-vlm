from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ...models.exact_speculative_verify import exact_speculative_verify_dense_available
from ...models.exact_speculative_verify import (
    exact_speculative_verify_weight as _target_verify_weight,
)
from ...models.mxfp4_verifier import (
    optimized_mxfp4_argmax as _target_verify_optimized_mxfp4_argmax,
)
from ...models.mxfp4_verifier import (
    optimized_mxfp4_linear as _target_verify_optimized_mxfp4_linear,
)
from ...models.mxfp4_verifier import (
    optimized_mxfp4_linears as _target_verify_optimized_mxfp4_linears,
)
from ...models.quantized_verifier import (
    optimized_affine_argmax as _target_verify_optimized_affine_argmax,
)
from ...models.quantized_verifier import (
    optimized_affine_linear as _target_verify_optimized_affine_linear,
)
from ...models.quantized_verifier import (
    optimized_affine_linears as _target_verify_optimized_affine_linears,
)
from ...models.quantized_verifier import (
    optimized_nvfp4_argmax as _target_verify_optimized_nvfp4_argmax,
)
from ...models.quantized_verifier import (
    singleton_quantized_argmax as _target_verify_singleton_quantized_argmax,
)
from ...models.quantized_verifier import (
    singleton_quantized_linear as _target_verify_singleton_quantized_linear,
)
from ...models.quantized_verifier import supports_quantization


def _use_target_verify_dense(linear, x: mx.array) -> bool:
    return (
        exact_speculative_verify_dense_available()
        and x.ndim == 3
        and (x.shape[0] > 1 or x.shape[1] > 1)
        and isinstance(linear, (nn.Linear, nn.QuantizedLinear))
    )


def _target_verify_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    output = _target_verify_optimized_mxfp4_linear(linear, x)
    if output is not None:
        return output
    output = _target_verify_optimized_affine_linear(linear, x)
    if output is not None:
        return output
    return _target_verify_singleton_quantized_linear(linear, x)


def _target_verify_quantized_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    output = _target_verify_optimized_mxfp4_argmax(linear, x, token_mask=token_mask)
    if output is not None:
        return output
    output = _target_verify_optimized_affine_argmax(linear, x, token_mask=token_mask)
    if output is not None:
        return output
    output = _target_verify_optimized_nvfp4_argmax(linear, x, token_mask=token_mask)
    if output is not None:
        return output
    return _target_verify_singleton_quantized_argmax(linear, x, token_mask=token_mask)


def _can_target_verify_quantized_head(linear) -> bool:
    return supports_quantization(linear)


def _target_verify_timewise(fn, x: mx.array) -> mx.array:
    return mx.concatenate(
        [fn(mx.contiguous(x[:, i : i + 1])) for i in range(x.shape[1])], axis=1
    )


def _target_verify_singletons(fn, x: mx.array) -> mx.array:
    rows = []
    for row in range(x.shape[0]):
        rows.append(
            mx.concatenate(
                [fn(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])],
                axis=1,
            )
        )
    return mx.concatenate(rows, axis=0)


def _target_verify_linear(linear, x: mx.array) -> mx.array:
    if not _use_target_verify_dense(linear, x):
        return linear(x)

    if isinstance(linear, nn.QuantizedLinear):
        out = _target_verify_quantized_linear(linear, x)
        if out is not None:
            return out
        if x.shape[0] > 1:
            return _target_verify_singletons(linear, x)
        return _target_verify_timewise(linear, x)

    if isinstance(linear, nn.Linear) and "bias" not in linear:
        out = _target_verify_weight(linear.weight, x)
        if out is not None:
            return out

    return _target_verify_singletons(linear, x)


def _target_verify_linears(linears, x: mx.array):
    if not (
        x.ndim == 3
        and (x.shape[0] > 1 or x.shape[1] > 1)
        and all(
            isinstance(linear, (nn.Linear, nn.QuantizedLinear)) for linear in linears
        )
    ):
        out = _decode_quantized_linears_fused(linears, x)
        if out is not None:
            return out
        return tuple(linear(x) for linear in linears)

    out = _target_verify_optimized_mxfp4_linears(linears, x)
    if out is not None:
        return out
    out = _target_verify_optimized_affine_linears(linears, x)
    if out is not None:
        return out
    return tuple(_target_verify_linear(linear, x) for linear in linears)


def _target_verify_embedding_as_linear(embedding, x: mx.array):
    if not (x.ndim == 3 and (x.shape[0] > 1 or x.shape[1] > 1)):
        return embedding.as_linear(x)

    out = _target_verify_weight(embedding.weight, x)
    if out is not None:
        return out

    return _target_verify_timewise(embedding.as_linear, x)


def _decode_quantized_linears_fused(linears, x: mx.array):
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or len(linears) != 4
        or not all(isinstance(linear, nn.QuantizedLinear) for linear in linears)
    ):
        return None

    first = linears[0]
    if not all(
        linear.bits == first.bits
        and linear.group_size == first.group_size
        and linear.mode == first.mode
        and linear.biases is not None
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and "bias" not in linear
        for linear in linears
    ):
        return None

    cache_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases)) for linear in linears
    )
    cached = getattr(first, "_fused_decode_linears", None)
    if cached is None or cached[0] != cache_key:
        weights = mx.concatenate([linear.weight for linear in linears], axis=0)
        scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        biases = mx.concatenate([linear.biases for linear in linears], axis=0)
        split_indices = []
        offset = 0
        for linear in linears[:-1]:
            offset += linear.weight.shape[0]
            split_indices.append(offset)
        mx.eval(weights, scales, biases)
        cached = (cache_key, weights, scales, biases, split_indices)
        first._fused_decode_linears = cached

    _, weights, scales, biases, split_indices = cached
    output = mx.quantized_matmul(
        x,
        weights,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=first.group_size,
        bits=first.bits,
        mode=first.mode,
    )
    return tuple(mx.split(output, split_indices, axis=-1))
