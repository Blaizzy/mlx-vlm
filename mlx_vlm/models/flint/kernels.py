"""Inference kernels preserving Flint's FP32 expert weighting."""

from functools import lru_cache, partial

import mlx.core as mx

from ..fast_ops import _affine_exact_header
from ..switch_layers import QuantizedSwitchLinear, SwiGLU


@partial(mx.compile, shapeless=True)
def normalize_expert_weights(scores, indices):
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)


@lru_cache(None)
def _route_kernel():
    return mx.fast.metal_kernel(
        name="flint_route_six",
        input_names=["scores", "choice"],
        output_names=["indices", "weights"],
        header="#pragma clang fp contract(off)\n",
        source="""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        float values[8];
        bool valid[8];
        int selected[6];
        for (int j = 0; j < 8; ++j) {
            values[j] = choice[row * 256 + j * 32 + lane];
            valid[j] = true;
        }
        // Match the stable ascending sort used by MLX argpartition, including
        // ties, infinities and NaNs. Invalid entries cannot be selected again.
        for (int k = 5; k >= 0; --k) {
            float best = -INFINITY;
            int nan_index = -1;
            for (int j = 0; j < 8; ++j) {
                if (!valid[j]) continue;
                if (isnan(values[j])) nan_index = j * 32 + lane;
                else best = max(best, values[j]);
            }
            best = simd_max(best);
            nan_index = simd_max(nan_index);
            int index = -1;
            for (int j = 0; j < 8; ++j) {
                if (valid[j] && values[j] == best) index = j * 32 + lane;
            }
            index = nan_index >= 0 ? nan_index : simd_max(index);
            selected[k] = index;
            for (int j = 0; j < 8; ++j) {
                if (j * 32 + lane == index) valid[j] = false;
            }
        }
        if (lane == 0) {
            float total = 0.0f;
            for (int k = 0; k < 6; ++k) total += scores[row * 256 + selected[k]];
            total += 1e-20f;
            for (int k = 0; k < 6; ++k) {
                indices[row * 6 + k] = selected[k];
                weights[row * 6 + k] = scores[row * 256 + selected[k]] / total;
            }
        }
        """,
    )


def route_experts(scores, choice, top_k):
    if (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and top_k == 6
        and scores.shape[-1] == 256
        and scores.shape == choice.shape
        and scores.dtype == choice.dtype == mx.float32
        and scores.size > 0
    ):
        shape = (*scores.shape[:-1], top_k)
        return _route_kernel()(
            inputs=[scores, choice],
            grid=(32 * (scores.size // 256), 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[mx.int32, mx.float32],
        )
    indices = mx.stop_gradient(
        mx.argpartition(choice, kth=-top_k, axis=-1)[..., -top_k:]
    )
    return indices, normalize_expert_weights(scores, indices)


def weighted_sum(routed, weights):
    return (
        (routed.astype(mx.float32) * weights[..., None])
        .sum(axis=-2)
        .astype(routed.dtype)
    )


@lru_cache(None)
def _weighted_kernel():
    return mx.fast.metal_kernel(
        name="flint_weighted_sum",
        input_names=["routed", "weights", "shared", "limit"],
        output_names=["out"],
        header="#pragma clang fp contract(off)\n",
        source="""
        uint i = thread_position_in_grid.x;
        if (i >= limit) return;
        uint row = i / D;
        uint col = i % D;
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            float term = float(routed[(row * K + k) * D + col]) * weights[row * K + k];
            value = value + term;
        }
        T result = T(value);
        if (HAS_SHARED) result = T(result + shared[i]);
        out[i] = result;
        """,
    )


def metal_sum(routed, weights, shared=None):
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or routed.dtype not in (mx.bfloat16, mx.float16)
        or weights.dtype != mx.float32
        or routed.shape[-2] != 6
        or routed.size == 0
        or (
            shared is not None
            and (
                shared.dtype != routed.dtype
                or shared.shape != (*routed.shape[:-2], routed.shape[-1])
            )
        )
    ):
        output = weighted_sum(routed, weights)
        return output if shared is None else output + shared
    shape = (*routed.shape[:-2], routed.shape[-1])
    size = routed.size // routed.shape[-2]
    return _weighted_kernel()(
        inputs=[
            routed,
            weights,
            shared if shared is not None else routed,
            mx.array(size, mx.uint32),
        ],
        template=[
            ("T", routed.dtype),
            ("D", routed.shape[-1]),
            ("K", routed.shape[-2]),
            ("HAS_SHARED", shared is not None),
        ],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[routed.dtype],
    )[0]


def supported(linear, x):
    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and type(linear) is QuantizedSwitchLinear
        and linear.bits == 4
        and linear.group_size == 64
        and linear.mode == "affine"
        and "bias" not in linear
        and linear.biases is not None
        and x.dtype in (mx.bfloat16, mx.float16)
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and linear.input_dims == x.shape[-1]
        and linear.input_dims % 512 == 0
        and linear.output_dims % 8 == 0
    )


@lru_cache(None)
def gate_kernel():
    source = r"""
    uint lane = thread_index_in_simdgroup;
    int route = threadgroup_position_in_grid.z;
    int row0 = threadgroup_position_in_grid.y * ROWS_PER_TG +
        simdgroup_index_in_threadgroup * RESULTS_PER_SIMDGROUP;
    int expert = int(indices[route]);
    constexpr int ROW_BYTES = K_SIZE / 2;
    constexpr int GROUPS = K_SIZE / GROUP_SIZE;
    const device T* xp = x + (route / TOP_K) * K_SIZE + lane * VALUES_PER_THREAD;
    int woff = expert * N_SIZE * ROW_BYTES + row0 * ROW_BYTES +
        lane * PACKS_PER_THREAD * BYTES_PER_PACK;
    int soff = expert * N_SIZE * GROUPS + row0 * GROUPS +
        lane / SCALE_STEP_PER_THREAD;
    float up_result[RESULTS_PER_SIMDGROUP] = {0};
    float gate_result[RESULTS_PER_SIMDGROUP] = {0};
    float xt[VALUES_PER_THREAD];
    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sum = load_affine4_vector_exact<T>(xp, xt);
      for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
        up_result[r] += affine4_qdot_exact(
            (const device uint8_t*)up_w + woff + r * ROW_BYTES, xt,
            float(up_scales[soff + r * GROUPS]),
            float(up_biases[soff + r * GROUPS]), sum);
        gate_result[r] += affine4_qdot_exact(
            (const device uint8_t*)gate_w + woff + r * ROW_BYTES, xt,
            float(gate_scales[soff + r * GROUPS]),
            float(gate_biases[soff + r * GROUPS]), sum);
      }
      xp += BLOCK_SIZE;
      woff += BLOCK_SIZE / 2;
      soff += BLOCK_SIZE / GROUP_SIZE;
    }
    for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
      T up = T(simd_sum(up_result[r]));
      T gate = T(simd_sum(gate_result[r]));
      if (lane == 0) {
        // Match MLX Sigmoid and both BF16 rounding points in compiled SwiGLU.
        auto z = 1 / (1 + metal::precise::exp(metal::abs(gate)));
        T sigmoid = gate < 0 ? z : 1 - z;
        T silu = T(gate * sigmoid);
        y[route * N_SIZE + row0 + r] = T(silu * up);
      }
    }
    """
    return mx.fast.metal_kernel(
        name="flint_gate_up_silu",
        input_names=[
            "x",
            "indices",
            "up_w",
            "up_scales",
            "up_biases",
            "gate_w",
            "gate_scales",
            "gate_biases",
        ],
        output_names=["y"],
        header=_affine_exact_header(4, 64, 2, 1),
        source=source,
    )


def gate_up_silu(switch, x, indices):
    up, gate = getattr(switch, "up_proj", None), getattr(switch, "gate_proj", None)
    if (
        type(getattr(switch, "activation", None)) is not SwiGLU
        or x.ndim != 3
        or indices.shape[:2] != x.shape[:2]
        or indices.size >= 64
        or indices.shape[-1] != 6
        or not (supported(up, x) and supported(gate, x))
        or up.weight.shape != gate.weight.shape
    ):
        return None
    b, l, d = x.shape
    k = indices.shape[-1]
    n = up.output_dims
    return gate_kernel()(
        inputs=[
            x,
            indices.astype(mx.int32),
            up.weight,
            up.scales,
            up.biases,
            gate.weight,
            gate.scales,
            gate.biases,
        ],
        template=[
            ("T", x.dtype),
            ("K_SIZE", d),
            ("N_SIZE", n),
            ("TOP_K", k),
            ("GROUP_SIZE", 64),
        ],
        grid=(32, n // 2, b * l * k),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, l, k, n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(None)
def down_kernel():
    source = r"""
    uint lane = thread_index_in_simdgroup;
    uint route = simdgroup_index_in_threadgroup;
    int token = threadgroup_position_in_grid.z;
    int row = threadgroup_position_in_grid.y;
    constexpr int ROW_BYTES = K_SIZE / 2;
    constexpr int GROUPS = K_SIZE / GROUP_SIZE;
    threadgroup float projected[TOP_K];
    int slot = token * TOP_K + route;
    int expert = int(indices[slot]);
    const device T* xp = x + slot * K_SIZE + lane * VALUES_PER_THREAD;
    int woff = expert * N_SIZE * ROW_BYTES + row * ROW_BYTES +
        lane * PACKS_PER_THREAD * BYTES_PER_PACK;
    int soff = expert * N_SIZE * GROUPS + row * GROUPS +
        lane / SCALE_STEP_PER_THREAD;
    float result = 0.0f;
    float xt[VALUES_PER_THREAD];
    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
        float sum = load_affine4_vector_exact<T>(xp, xt);
        result += affine4_qdot_exact(
            (const device uint8_t*)w + woff, xt,
            float(scales[soff]), float(biases[soff]), sum);
        xp += BLOCK_SIZE;
        woff += BLOCK_SIZE / 2;
        soff += BLOCK_SIZE / GROUP_SIZE;
    }
    float value = float(T(simd_sum(result)));
    if (lane == 0) projected[route] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (route == 0 && lane == 0) {
        // Parallel projections retain the original ordered FP32 reduction.
        float total = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            float term = projected[k] * route_weights[token * TOP_K + k];
            total = total + term;
        }
        int pos = token * N_SIZE + row;
        y[pos] = T(T(total) + shared_y[pos]);
    }
    """
    return mx.fast.metal_kernel(
        name="flint_down_fp32_reduce",
        input_names=[
            "x",
            "indices",
            "route_weights",
            "w",
            "scales",
            "biases",
            "shared_y",
        ],
        output_names=["y"],
        header=_affine_exact_header(4, 64, 1, 6) + "\n#pragma clang fp contract(off)\n",
        source=source,
    )


def down_reduce(linear, x, indices, weights, shared):
    if (
        x.ndim != 4
        or indices.shape != x.shape[:3]
        or weights.shape != indices.shape
        or weights.dtype != mx.float32
        or indices.size >= 64
        or indices.shape[-1] != 6
        or not supported(linear, x)
        or shared.shape != (*x.shape[:2], linear.output_dims)
        or shared.dtype != x.dtype
    ):
        return None
    b, l, k, d = x.shape
    n = linear.output_dims
    return down_kernel()(
        inputs=[
            x,
            indices.astype(mx.int32),
            weights,
            linear.weight,
            linear.scales,
            linear.biases,
            shared,
        ],
        template=[
            ("T", x.dtype),
            ("K_SIZE", d),
            ("N_SIZE", n),
            ("TOP_K", k),
            ("GROUP_SIZE", 64),
        ],
        grid=(32, 6 * n, b * l),
        threadgroup=(32, 6, 1),
        output_shapes=[(b, l, n)],
        output_dtypes=[x.dtype],
    )[0]
