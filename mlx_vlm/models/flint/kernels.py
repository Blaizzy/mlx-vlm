"""Inference kernels preserving Flint's FP32 expert weighting."""

from functools import lru_cache

import mlx.core as mx

from ..fast_ops import _affine_exact_header
from ..switch_layers import QuantizedSwitchLinear


def weighted_sum(routed, weights):
    return (
        (routed.astype(mx.float32) * weights[..., None])
        .sum(axis=-2)
        .astype(routed.dtype)
    )


_weighted_kernel = mx.fast.metal_kernel(
    name="flint_weighted_sum",
    input_names=["routed", "weights"],
    output_names=["out"],
    header="#pragma clang fp contract(off)\n",
    source="""
    uint i = thread_position_in_grid.x;
    if (i >= SIZE) return;
    uint row = i / D;
    uint col = i % D;
    float value = 0.0f;
    for (int k = 0; k < K; ++k) {
        float term = float(routed[(row * K + k) * D + col]) * weights[row * K + k];
        value = value + term;
    }
    out[i] = T(value);
    """,
)


def metal_sum(routed, weights):
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or routed.dtype not in (mx.bfloat16, mx.float16)
        or weights.dtype != mx.float32
        or routed.shape[-2] != 6
        or routed.size == 0
    ):
        return weighted_sum(routed, weights)
    shape = (*routed.shape[:-2], routed.shape[-1])
    size = routed.size // routed.shape[-2]
    return _weighted_kernel(
        inputs=[routed, weights],
        template=[
            ("T", routed.dtype),
            ("D", routed.shape[-1]),
            ("K", routed.shape[-2]),
            ("SIZE", size),
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
        and isinstance(linear, QuantizedSwitchLinear)
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
        header=_affine_exact_header(4, 64, 4, 2),
        source=source,
    )


def gate_up_silu(switch, x, indices):
    up, gate = switch.up_proj, switch.gate_proj
    if (
        x.ndim != 3
        or x.shape[1] != 1
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
            ("VERIFY_T", l),
            ("K_SIZE", d),
            ("N_SIZE", n),
            ("TOP_K", k),
            ("GROUP_SIZE", 64),
        ],
        grid=(32, 2 * (n // 8), b * l * k),
        threadgroup=(32, 2, 1),
        output_shapes=[(b, l, k, n)],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(None)
def down_kernel():
    source = r"""
    uint lane = thread_index_in_simdgroup;
    int token = threadgroup_position_in_grid.z;
    int row0 = threadgroup_position_in_grid.y * ROWS_PER_TG +
        simdgroup_index_in_threadgroup * RESULTS_PER_SIMDGROUP;
    constexpr int ROW_BYTES = K_SIZE / 2;
    constexpr int GROUPS = K_SIZE / GROUP_SIZE;
    float routed_sum[RESULTS_PER_SIMDGROUP] = {0};
    for (int route = 0; route < TOP_K; ++route) {
      int slot = token * TOP_K + route;
      int expert = int(indices[slot]);
      const device T* xp = x + slot * K_SIZE + lane * VALUES_PER_THREAD;
      int woff = expert * N_SIZE * ROW_BYTES + row0 * ROW_BYTES +
          lane * PACKS_PER_THREAD * BYTES_PER_PACK;
      int soff = expert * N_SIZE * GROUPS + row0 * GROUPS +
          lane / SCALE_STEP_PER_THREAD;
      float result[RESULTS_PER_SIMDGROUP] = {0};
      float xt[VALUES_PER_THREAD];
      for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
        float sum = load_affine4_vector_exact<T>(xp, xt);
        for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
          result[r] += affine4_qdot_exact(
              (const device uint8_t*)w + woff + r * ROW_BYTES, xt,
              float(scales[soff + r * GROUPS]),
              float(biases[soff + r * GROUPS]), sum);
        }
        xp += BLOCK_SIZE;
        woff += BLOCK_SIZE / 2;
        soff += BLOCK_SIZE / GROUP_SIZE;
      }
      for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
        float projected = float(T(simd_sum(result[r])));
        if (lane == 0) {
          float product = projected * route_weights[slot];
          routed_sum[r] = routed_sum[r] + product;
        }
      }
    }
    if (lane == 0) {
      for (int r = 0; r < RESULTS_PER_SIMDGROUP; ++r) {
        int pos = token * N_SIZE + row0 + r;
        // Cast after FP32 reduction, then add the shared expert in T.
        y[pos] = T(T(routed_sum[r]) + shared_y[pos]);
      }
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
        header=_affine_exact_header(4, 64, 2, 4) + "\n#pragma clang fp contract(off)\n",
        source=source,
    )


def down_reduce(linear, x, indices, weights, shared):
    if (
        x.ndim != 4
        or x.shape[1] != 1
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
            ("VERIFY_T", l),
            ("K_SIZE", d),
            ("N_SIZE", n),
            ("TOP_K", k),
            ("GROUP_SIZE", 64),
        ],
        grid=(32, n // 2, b * l),
        threadgroup=(32, 4, 1),
        output_shapes=[(b, l, n)],
        output_dtypes=[x.dtype],
    )[0]
