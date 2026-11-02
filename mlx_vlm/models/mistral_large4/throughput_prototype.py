"""Opt-in two-Mac verifier: split dense output rows and batch cache updates.

The prototype retains complete weights on each rank. It partitions only dense
output rows, gathers their values, and keeps every scalar dot's reduction order.
"""

from contextlib import contextmanager
from functools import lru_cache, partial

import mlx.core as mx

from . import moe_kernel as mk
from . import speculative_prototype as proto
from . import speculative_verifier as verifier
from . import verify_kernel as vk
from .language import MoE
from .next_prototype import attention_block


@lru_cache(maxsize=1)
def _kernel():
    header = vk._HEADER.replace(
        "typename AccT = float>", "typename AccT = float, const int SG=2>"
    ).replace("num_simdgroups = k_lanes / 8", "num_simdgroups = SG")
    return mx.fast.metal_kernel(
        name="ml4_throughput_dense",
        input_names=["x", "weight"],
        output_names=["out"],
        header=header,
        source="""GemvWide<T,V,KL,float,SG>::run(weight,x,weight,out,x_shape[2],weight_shape[0],x_shape[0]*x_shape[1],x_shape[2],x_shape[2],1.f,0.f,0,0,threadgroup_position_in_grid,threadgroups_per_grid,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);""",
    )


def _tuned_linear(weight, x):
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.ndim != 3
        or not 2 <= x.shape[0] <= 15
        or x.dtype not in (mx.bfloat16, mx.float16)
        or weight.dtype != x.dtype
        or x.shape[-1] % 4
        or weight.shape[0] <= 1
    ):
        return None
    generation = mx.device_info().get("architecture", "").split("_g")[-1][:2]
    if not generation.isdigit() or int(generation) < 15:
        return None
    if x.shape[1] == 1:
        return vk.native_wide_linear(weight, x)
    b = x.shape[0]
    n, k = weight.shape
    lanes = 32 if b <= 5 or n <= 64 else 16
    if n >= 49152:
        v, c, sg = (8, 2, lanes // 8) if b >= 6 else (8, 1, lanes // 8)
    elif b >= 6:
        v, c, sg = 8, 4, 8
    elif n <= 1024:
        v, c, sg = 4, 4, 1
    elif k >= 16384:
        v, c, sg = 8, (4 if b >= 5 else 2), lanes // 8
    elif n >= 12288:
        v, c, sg = 8, 4, 8
    else:
        v, c, sg = 4, 4, 4
    rows = sg * 32 // lanes
    return _kernel()(
        inputs=[x, weight],
        template=[("T", x.dtype), ("KL", lanes), ("V", v), ("SG", sg)],
        grid=(32 * c, ((n + rows - 1) // rows) * sg, 1),
        threadgroup=(32, sg, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]


def _factor(linear):
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
def _local_experts(x, gate_weight, packed, scales, factors, start, routed_scaling):
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
    outputs = project(proto.activation(gate, up), 2)[inverse].reshape(
        *indices.shape, x.shape[-1]
    )
    return outputs, weights


@contextmanager
def enabled(group=None, *, model=None, compile_experts=False, evaluation_interval=4):
    """Process-wide experiment switch; keep it fixed throughout generation."""
    size = group.size() if group is not None else 1
    rank = group.rank() if group is not None else 0
    parameters = {}
    if compile_experts:
        if model is None:
            raise ValueError("model is required for compiled experts")
        for layer in model.layers:
            ffn = layer.feed_forward
            if (
                not isinstance(ffn, MoE)
                or ffn.group is None
                or ffn.args.n_group != 1
                or ffn.args.num_experts_per_tok != 4
            ):
                continue
            modules = (ffn.experts.w1, ffn.experts.w3, ffn.experts.w2)
            if not all(mk.supports_masked_qmv(m) for m in modules):
                continue
            parameters[id(ffn)] = (
                [m.weight for m in modules],
                [m.scales for m in modules],
                [_factor(m) for m in modules],
            )
        mx.eval([p[2] for p in parameters.values()])

    def dense(weight, x):
        n, k = weight.shape
        if (
            size < 2
            or x.ndim != 3
            or x.shape[0] <= 1
            or n < 6144
            or k < 6144
            or n % size
        ):
            return _tuned_linear(weight, x)
        rows = n // size
        out = _tuned_linear(weight[rank * rows : (rank + 1) * rows], x)
        if out is None:
            return None
        return (
            mx.distributed.all_gather(out.reshape(1, -1, rows), group=group)
            .transpose(1, 0, 2)
            .reshape(*x.shape[:-1], n)
        )

    def feed_forward(ffn, x):
        if id(ffn) not in parameters:
            return proto.feed_forward(ffn, x)
        packed, scales, factors = parameters[id(ffn)]
        outputs, weights = _local_experts(
            x,
            ffn.gate.weight,
            packed,
            scales,
            factors,
            ffn.start,
            ffn.args.routed_scaling_factor,
        )
        result = proto.combine(
            mx.distributed.all_sum(outputs, group=ffn.group), weights
        )
        if ffn.shared_experts is not None:
            result = result + verifier.dense_mlp(ffn.shared_experts, x)
        return result

    old_linear, old_hidden = verifier.native_wide_linear, verifier.verify_hidden
    with proto.enabled(block=True, compiled=True):
        verifier.native_wide_linear = dense
        verifier.attention_block = attention_block
        verifier.verify_hidden = partial(
            old_hidden, evaluation_interval=evaluation_interval
        )
        if compile_experts:
            verifier.feed_forward = feed_forward
        try:
            yield
        finally:
            verifier.native_wide_linear, verifier.verify_hidden = old_linear, old_hidden
