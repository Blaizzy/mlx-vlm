"""Compiled arithmetic for Mistral speculative verification."""

import mlx.core as mx
import mlx.nn as nn

from . import moe_kernel as mk


@mx.compile
def activation(gate, up):
    return nn.silu(gate) * up


@mx.compile
def combine(outputs, weights):
    weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(outputs.dtype)
    result = weighted[..., 0, :]
    for i in range(1, weights.shape[-1]):
        result = result + weighted[..., i, :]
    return result


def expert_factor(linear):
    e = linear.num_experts
    if hasattr(linear, "global_scale"):
        scale = linear.global_scale.astype(mx.float32)
        factor = (
            2688.0 / scale
            if linear._scale_name == "weight_global_scale"
            else 2688.0 * scale
        )
        factor = mx.broadcast_to(factor.reshape(-1), (e,))
    else:
        factor = mx.full((e,), 2688.0, mx.float32)
    if e < 64:
        factor = mx.concatenate([factor, mx.zeros((64 - e,), mx.float32)])
    return factor


@mx.compile
def local_experts(x, gate_weight, packed, scales, factors, start, routed_scaling):
    all_indices = []
    all_weights = []
    for t in range(x.shape[1]):
        scores = mx.softmax(
            mx.contiguous(x[:, t : t + 1]).astype(mx.float32)
            @ gate_weight.T.astype(mx.float32),
            axis=-1,
            precise=True,
        )
        ids = mx.argpartition(-scores, kth=3, axis=-1)[..., :4]
        ids = mx.stop_gradient(mx.sort(ids, axis=-1))
        weights = mx.take_along_axis(scores, ids, axis=-1)
        weights = (
            weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * routed_scaling
        )
        all_indices.append(ids)
        all_weights.append(weights)
    indices = mx.concatenate(all_indices, axis=1)
    weights = mx.concatenate(all_weights, axis=1)
    experts = packed[0].shape[0]
    local = mx.where(
        (indices >= start) & (indices < start + experts),
        indices.astype(mx.int32) - start,
        -1,
    )
    order = mx.argsort(local.reshape(-1))
    inverse = mx.argsort(order)
    rx = x.reshape(-1, x.shape[-1])[order // local.shape[-1], None]
    ri = local.reshape(-1)[order, None]

    def project(value, i):
        k = value.shape[-1]
        n = packed[i].shape[1]
        return mk._masked_qmv()(
            inputs=[value, packed[i], scales[i], factors[i], ri],
            template=[("T", value.dtype), ("K", k), ("N", n), ("ROUTES_PER_X", 1)],
            grid=(32, ((n + 7) // 8) * 2, ri.size),
            threadgroup=(32, 2, 1),
            output_shapes=[(*ri.shape, n)],
            output_dtypes=[value.dtype],
        )[0]

    gate, up = project(rx, 0), project(rx, 1)
    outputs = project(activation(gate, up), 2)[inverse].reshape(
        *indices.shape, x.shape[-1]
    )
    return outputs, weights
