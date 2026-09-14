"""Format-independent quantized operations for exact speculative verification."""

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .switch_layers import QuantizedSwitchLinear

AFFINE_BITS = frozenset((2, 3, 4, 5, 6, 8))
FIXED_FORMATS = {
    "mxfp4": (4, 32),
    "mxfp8": (8, 32),
    "nvfp4": (4, 16),
}

_QUANTIZED_MOE_HC_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
"""


_AFFINE_MOE_HC_HEADER = _QUANTIZED_MOE_HC_HEADER + r"""
constant constexpr int PACK_FACTOR =
    (BITS == 3 || BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS);
constant constexpr int BYTES_PER_PACK =
    ((BITS & (BITS - 1)) == 0) ? 4 : (BITS == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = BITS == 2 ? 1 : 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
constant constexpr int SCALE_STEP_PER_THREAD =
    GROUP_SIZE / VALUES_PER_THREAD;
constant constexpr bool HAS_AFFINE_BIAS = true;

template <typename T>
inline float load_quantized_vector(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  if constexpr (BITS == 3 || BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
    }
  } else if constexpr (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      sum += x[i];
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    }
  }
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    int shift = (i * BITS) % 8;
    x_thread[i] = x[i] / float(1 << shift);
  }
  return sum;
}

inline float quantized_dot(
    const device uint8_t* w,
    const thread float* x,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  constexpr uint MASK = (1u << BITS) - 1;
  if constexpr (BITS == 2 || BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      float partial = 0.0f;
      for (int element = 0; element < 4; ++element) {
        int value_index = i + element;
        int bit = value_index * BITS;
        int byte = bit / 8;
        int shift = bit % 8;
        partial += x[value_index] * (w[byte] & ((MASK << shift) & 0xff));
      }
      accum += partial;
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      int bit = i * BITS;
      int byte = bit / 8;
      int shift = bit % 8;
      accum += x[i] * (w[byte] & ((MASK << shift) & 0xff));
      if (shift + BITS > 8) {
        uint high_mask = MASK >> (8 - shift);
        accum += (w[byte + 1] & high_mask) * (x[i] * 256.0f);
      }
    }
  }
  return scale * accum + sum * bias;
}

template <typename T>
inline float quantized_scale(T value) {
  return float(value);
}
"""


_FP_MOE_HC_HEADER = _QUANTIZED_MOE_HC_HEADER + r"""
constant constexpr int PACK_FACTOR = 32 / BITS;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
constant constexpr int SCALE_STEP_PER_THREAD =
    GROUP_SIZE / VALUES_PER_THREAD;
constant constexpr bool HAS_AFFINE_BIAS = false;

inline float decode_fp8_e4m3(uint8_t value) {
  half decoded = as_type<half>(ushort((value & 127) << 7));
  decoded *= 256.0f;
  return float(value & 128 ? -decoded : decoded);
}

inline float decode_fp8_e8m0(uint8_t value) {
  uint32_t decoded = value == 0 ? 0x400000 : uint32_t(value) << 23;
  return as_type<float>(decoded);
}

inline float decode_fp4_e2m1(uint8_t value) {
  half decoded = as_type<half>(ushort((value & 7) << 9));
  decoded *= 16384.0f;
  return float(value & 8 ? -decoded : decoded);
}

template <typename T>
inline float load_quantized_vector(
    const device T* x,
    thread float* x_thread) {
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    x_thread[i] = x[i];
  }
  return 0.0f;
}

inline float quantized_dot(
    const device uint8_t* w,
    const thread float* x,
    float scale,
    float,
    float) {
  float accum = 0.0f;
  if constexpr (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
      accum += x[4 * i] * decode_fp4_e2m1(ws[i]) +
          x[4 * i + 1] * decode_fp4_e2m1(ws[i] >> 4) +
          x[4 * i + 2] * decode_fp4_e2m1(ws[i] >> 8) +
          x[4 * i + 3] * decode_fp4_e2m1(ws[i] >> 12);
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      accum += x[i] * decode_fp8_e4m3(w[i]);
    }
  }
  return scale * accum;
}

inline float quantized_scale(uint8_t value) {
  if constexpr (GROUP_SIZE == 16) {
    return decode_fp8_e4m3(value);
  }
  return decode_fp8_e8m0(value);
}
"""


_FUSED_QUANTIZED_MOE_HC_SOURCE = r"""
uint n_tile = threadgroup_position_in_grid.y;
uint token = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

constexpr int RESULTS_PER_SIMDGROUP = 2;
constexpr int NUM_SIMDGROUPS = 4;
constexpr int ROWS_PER_TG = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
constexpr int W_ROW_BYTES = K_SIZE * BITS / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;
int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;

T routed_sum[RESULTS_PER_SIMDGROUP] = {T(0)};
for (int route = 0; route < TOP_K; ++route) {
  int route_offset = int(token) * TOP_K + route;
  int expert = int(indices[route_offset]);
  const device uint8_t* ws = (const device uint8_t*)w +
      expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
      int(lane) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device auto* sc = scales + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(lane) / SCALE_STEP_PER_THREAD;
  const device T* bs = biases;
  if constexpr (HAS_AFFINE_BIAS) {
    bs += expert * S_EXPERT_SIZE + out_row * GROUPS +
        int(lane) / SCALE_STEP_PER_THREAD;
  }
  const device T* xk = x + route_offset * K_SIZE +
      int(lane) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];
  for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
    float sum = load_quantized_vector<T>(xk, x_thread);
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      const device uint8_t* wr = ws + row * W_ROW_BYTES;
      const device auto* sr = sc + row * GROUPS;
      const device T* br = bs + (HAS_AFFINE_BIAS ? row * GROUPS : 0);
      float bias = HAS_AFFINE_BIAS ? float(br[0]) : 0.0f;
      result[row] += quantized_dot(
          wr, x_thread, quantized_scale(sr[0]), bias, sum);
    }
    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GROUP_SIZE;
    if constexpr (HAS_AFFINE_BIAS) {
      bs += BLOCK_SIZE / GROUP_SIZE;
    }
    xk += BLOCK_SIZE;
  }

  T route_weight = T(route_weights[route_offset]);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    float value = simd_sum(result[row]);
    if (lane == 0) {
      T product = T(T(value) * route_weight);
      routed_sum[row] = T(routed_sum[row] + product);
    }
  }
}

threadgroup T collapsed_tile[ROWS_PER_TG];
for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  if (lane == 0) {
    collapsed_tile[int(simd_gid) * RESULTS_PER_SIMDGROUP + row] =
        T(routed_sum[row] + shared[int(token) * N_SIZE + n]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0) {
  short qid = lane / 4;
  short matrix_row = (qid & 4) + ((lane / 2) % 4);
  short matrix_col = (qid & 2) * 2 + (lane % 2) * 2;
  float2 a_values = 0.0f;
  float2 b_values = 0.0f;
  if (matrix_row < HC) {
    for (short element = 0; element < 2; ++element) {
      short source = matrix_col + element;
      if (source < HC) {
        a_values[element] = comb[
            int(token) * HC * HC + source * HC + matrix_row];
      }
      b_values[element] = float(residual[
          int(token) * HC * N_SIZE + matrix_row * N_SIZE +
          int(n_tile) * ROWS_PER_TG + matrix_col + element]);
    }
  }

  simdgroup_matrix<float, 8, 8> a_matrix;
  simdgroup_matrix<float, 8, 8> b_matrix;
  simdgroup_matrix<float, 8, 8> c_matrix;
  simdgroup_matrix<float, 8, 8> d_matrix;
  reinterpret_cast<thread float2&>(a_matrix.thread_elements()) = a_values;
  reinterpret_cast<thread float2&>(b_matrix.thread_elements()) = b_values;
  reinterpret_cast<thread float2&>(c_matrix.thread_elements()) = float2(0.0f);
  simdgroup_multiply_accumulate(d_matrix, a_matrix, b_matrix, c_matrix);

  float2 values =
      reinterpret_cast<thread float2&>(d_matrix.thread_elements());
  if (matrix_row < HC) {
    for (short element = 0; element < 2; ++element) {
      int column = int(n_tile) * ROWS_PER_TG + matrix_col + element;
      volatile float product = post[int(token) * HC + matrix_row] *
          float(collapsed_tile[matrix_col + element]);
      values[element] = product + values[element];
      out[int(token) * HC * N_SIZE + matrix_row * N_SIZE + column] =
          T(values[element]);
    }
  }
}
"""


_QUANTIZED_MOE_HC_SOURCE = r"""
uint group = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
constexpr uint TILES = D / 8;
uint flat_tile = group * SIMDS + simd_gid;
uint row = flat_tile / TILES;
uint tile = flat_tile - row * TILES;
if (row >= ROWS) {
  return;
}

short qid = lane / 4;
short matrix_row = (qid & 4) + ((lane / 2) % 4);
short matrix_col = (qid & 2) * 2 + (lane % 2) * 2;

float2 a_values = 0.0f;
float2 b_values = 0.0f;
if (matrix_row < HC) {
  for (short element = 0; element < 2; ++element) {
    short source = matrix_col + element;
    if (source < HC) {
      a_values[element] = comb[
          row * HC * HC + source * HC + matrix_row];
    }
    b_values[element] = float(residual[
        row * HC * D + matrix_row * D + tile * 8 + matrix_col + element]);
  }
}

simdgroup_matrix<float, 8, 8> a_matrix;
simdgroup_matrix<float, 8, 8> b_matrix;
simdgroup_matrix<float, 8, 8> c_matrix;
simdgroup_matrix<float, 8, 8> d_matrix;
reinterpret_cast<thread float2&>(a_matrix.thread_elements()) = a_values;
reinterpret_cast<thread float2&>(b_matrix.thread_elements()) = b_values;
reinterpret_cast<thread float2&>(c_matrix.thread_elements()) = float2(0.0f);
simdgroup_multiply_accumulate(d_matrix, a_matrix, b_matrix, c_matrix);

float2 values =
    reinterpret_cast<thread float2&>(d_matrix.thread_elements());
if (matrix_row < HC) {
  for (short element = 0; element < 2; ++element) {
    uint column = tile * 8 + matrix_col + element;
    T routed_sum = T(0);
    for (int route = 0; route < TOP_K; ++route) {
      int route_offset = (row * TOP_K + route) * D + column;
      T route_weight = T(route_weights[row * TOP_K + route]);
      T routed_product = T(routed[route_offset] * route_weight);
      routed_sum = T(routed_sum + routed_product);
    }
    T collapsed = T(routed_sum + shared[row * D + column]);
    volatile float product =
        post[row * HC + matrix_row] * float(collapsed);
    values[element] = product + values[element];
    out[row * HC * D + matrix_row * D + column] = T(values[element]);
  }
}
"""


def supports_quantization(module) -> bool:
    """Return whether ``module`` uses one of MLX's native weight formats."""
    if not isinstance(module, (nn.QuantizedLinear, QuantizedSwitchLinear)):
        return False
    if module.mode == "affine":
        return module.bits in AFFINE_BITS and module.group_size in (32, 64, 128)
    return (module.bits, module.group_size) == FIXED_FORMATS.get(module.mode)


def _input_dims(module) -> int:
    return int(module.scales.shape[-1]) * int(module.group_size)


def _exact_time_batch(linear: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """Run every verifier position with the target's native batch shape."""
    return mx.concatenate(
        [
            linear(mx.contiguous(x[:, position : position + 1]))
            for position in range(x.shape[1])
        ],
        axis=1,
    )


def exact_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    """General BxT quantized projection with decode-equivalent arithmetic."""
    if (
        not supports_quantization(linear)
        or not isinstance(linear, nn.QuantizedLinear)
        or x.ndim != 3
        or x.shape[-1] != _input_dims(linear)
    ):
        return None
    if x.shape[1] <= 1:
        return linear(x)
    return _exact_time_batch(linear, x)


def singleton_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Fuse a runtime-sized batch while retaining singleton qmv arithmetic."""
    if (
        not supports_quantization(linear)
        or not isinstance(linear, nn.QuantizedLinear)
        or x.ndim != 3
        or x.shape[-1] != _input_dims(linear)
    ):
        return None

    indices = mx.zeros((*x.shape[:2], 1), dtype=mx.int32)
    projected = mx.expand_dims(mx.contiguous(x), (-2, -3))
    output = (
        mx.gather_qmm(
            projected,
            linear.weight[None],
            linear.scales[None],
            None if linear.biases is None else linear.biases[None],
            rhs_indices=indices,
            transpose=True,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
            sorted_indices=False,
        )
        .squeeze(-2)
        .squeeze(-2)
    )
    if "bias" in linear:
        output = output + linear["bias"]
    return output


def exact_quantized_switch_linear(
    linear,
    x: mx.array,
    indices: mx.array,
) -> Optional[mx.array]:
    """Project one hidden vector through every selected expert exactly."""
    if (
        not supports_quantization(linear)
        or not isinstance(linear, QuantizedSwitchLinear)
        or x.ndim != 3
        or indices.ndim != 3
        or indices.shape[:2] != x.shape[:2]
        or x.shape[-1] != _input_dims(linear)
    ):
        return None
    projected = mx.expand_dims(mx.contiguous(x), (-2, -3))
    return linear(projected, indices, sorted_indices=False).squeeze(-2)


def exact_quantized_selected_linear(
    linear,
    x: mx.array,
    indices: mx.array,
) -> Optional[mx.array]:
    """Project per-expert hidden vectors without flattening verifier time."""
    if (
        not supports_quantization(linear)
        or not isinstance(linear, QuantizedSwitchLinear)
        or x.ndim != 4
        or indices.ndim != 3
        or indices.shape != x.shape[:3]
        or x.shape[-1] != _input_dims(linear)
    ):
        return None
    projected = mx.expand_dims(mx.contiguous(x), -2)
    return linear(projected, indices, sorted_indices=False).squeeze(-2)


@lru_cache(maxsize=None)
def _fused_quantized_moe_hc_kernel(
    mode,
    bits,
    group_size,
    dtype,
    length,
    input_size,
    output_size,
    top_k,
    hc_mult,
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unknown")
    header = _AFFINE_MOE_HC_HEADER if mode == "affine" else _FP_MOE_HC_HEADER
    header = header.replace("BITS", str(bits)).replace("GROUP_SIZE", str(group_size))
    return mx.fast.metal_kernel(
        name=(
            f"quantized_verify_fused_{mode}{bits}_moe_hc_{dtype_name}_"
            f"t{length}_k{input_size}_n{output_size}_e{top_k}_h{hc_mult}_"
            f"g{group_size}"
        ),
        input_names=[
            "x",
            "indices",
            "route_weights",
            "w",
            "scales",
            "biases",
            "shared",
            "residual",
            "post",
            "comb",
        ],
        output_names=["out"],
        header=header,
        source=_FUSED_QUANTIZED_MOE_HC_SOURCE,
    )


def _fused_quantized_moe_hc_expand(
    linear,
    x: mx.array,
    indices: mx.array,
    route_weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
) -> Optional[mx.array]:
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or not supports_quantization(linear)
        or not isinstance(linear, QuantizedSwitchLinear)
        or "bias" in linear
        or x.ndim != 4
        or x.shape[0] < 1
        or indices.shape != x.shape[:3]
        or route_weights.shape != indices.shape
        or route_weights.dtype != mx.float32
        or x.dtype not in (mx.bfloat16, mx.float16)
        or shared.shape != (x.shape[0], x.shape[1], linear.output_dims)
        or shared.dtype != x.dtype
        or residual.ndim != 4
        or residual.shape != (*x.shape[:2], residual.shape[2], linear.output_dims)
        or residual.dtype != x.dtype
        or post.shape != residual.shape[:-1]
        or comb.shape != (*residual.shape[:-1], residual.shape[2])
        or post.dtype != mx.float32
        or comb.dtype != mx.float32
        or residual.shape[2] > 8
    ):
        return None

    if linear.mode == "affine":
        if (
            linear.biases is None
            or linear.scales.dtype != x.dtype
            or linear.biases.dtype != x.dtype
        ):
            return None
    elif linear.scales.dtype != mx.uint8 or linear.biases is not None:
        return None

    batch, length, top_k, input_size = x.shape
    output_size = linear.output_dims
    pack_factor = (
        8 if linear.bits in (3, 5) else 4 if linear.bits == 6 else 32 // linear.bits
    )
    packs_per_thread = 1 if linear.mode == "affine" and linear.bits == 2 else 2
    block_size = pack_factor * packs_per_thread * 32
    if (
        input_size != linear.input_dims
        or input_size % block_size
        or output_size % 8
        or linear.group_size % (pack_factor * packs_per_thread)
    ):
        return None

    kernel = _fused_quantized_moe_hc_kernel(
        linear.mode,
        linear.bits,
        linear.group_size,
        x.dtype,
        length,
        input_size,
        output_size,
        top_k,
        residual.shape[2],
    )
    return kernel(
        inputs=[
            mx.contiguous(x),
            mx.contiguous(indices.astype(mx.int32)),
            mx.contiguous(route_weights),
            linear.weight,
            linear.scales,
            linear.biases if linear.biases is not None else shared,
            mx.contiguous(shared),
            mx.contiguous(residual),
            mx.contiguous(post),
            mx.contiguous(comb),
        ],
        template=[
            ("T", x.dtype),
            ("BITS", int(linear.bits)),
            ("GROUP_SIZE", int(linear.group_size)),
            ("K_SIZE", int(input_size)),
            ("N_SIZE", int(output_size)),
            ("TOP_K", int(top_k)),
            ("HC", int(residual.shape[2])),
        ],
        grid=(32, output_size // 2, batch * length),
        threadgroup=(32, 4, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(maxsize=None)
def _quantized_moe_hc_kernel(dtype, rows, hc_mult, width, top_k):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unknown")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_moe_hc_expand_"
            f"{dtype_name}_r{rows}_h{hc_mult}_d{width}_e{top_k}"
        ),
        input_names=["routed", "route_weights", "shared", "residual", "post", "comb"],
        output_names=["out"],
        header=_QUANTIZED_MOE_HC_HEADER,
        source=_QUANTIZED_MOE_HC_SOURCE,
    )


def exact_quantized_moe_hc_expand(
    linear,
    x: mx.array,
    indices: mx.array,
    route_weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
    *,
    routed: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Project quantized experts and apply one format-independent MoE/HC epilogue."""
    if routed is None:
        fused = _fused_quantized_moe_hc_expand(
            linear,
            x,
            indices,
            route_weights,
            shared,
            residual,
            post,
            comb,
        )
        if fused is not None:
            return fused
        routed = exact_quantized_selected_linear(linear, x, indices)
        if routed is None:
            return None
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or routed.ndim != 4
        or routed.dtype not in (mx.bfloat16, mx.float16)
        or route_weights.shape != routed.shape[:-1]
        or route_weights.dtype != mx.float32
        or shared.shape != (*routed.shape[:2], routed.shape[-1])
        or shared.dtype != routed.dtype
        or residual.shape[:2] != routed.shape[:2]
        or residual.shape[-1] != routed.shape[-1]
        or residual.dtype != routed.dtype
        or post.shape != residual.shape[:-1]
        or comb.shape != (*residual.shape[:-1], residual.shape[2])
        or post.dtype != mx.float32
        or comb.dtype != mx.float32
        or residual.shape[2] > 8
        or routed.shape[-1] % 8
    ):
        return None

    batch, length, top_k, width = routed.shape
    rows = batch * length
    simds = 8
    tiles = rows * (width // 8)
    kernel = _quantized_moe_hc_kernel(
        routed.dtype,
        rows,
        residual.shape[2],
        width,
        top_k,
    )
    return kernel(
        inputs=[
            mx.contiguous(routed),
            mx.contiguous(route_weights),
            mx.contiguous(shared),
            mx.contiguous(residual),
            mx.contiguous(post),
            mx.contiguous(comb),
        ],
        template=[
            ("T", routed.dtype),
            ("ROWS", int(rows)),
            ("HC", int(residual.shape[2])),
            ("D", int(width)),
            ("SIMDS", simds),
            ("TOP_K", int(top_k)),
        ],
        grid=(32 * ((tiles + simds - 1) // simds), simds, 1),
        threadgroup=(32, simds, 1),
        output_shapes=[residual.shape],
        output_dtypes=[routed.dtype],
    )[0]


def pad_token_mask(token_mask: mx.array, output_size: int) -> mx.array:
    """Pad a packed vocabulary mask to cover every output row."""
    required = (output_size + 31) // 32
    missing = required - token_mask.shape[-1]
    if missing <= 0:
        return token_mask
    shape = (*token_mask.shape[:-1], missing)
    return mx.concatenate([token_mask, mx.zeros(shape, token_mask.dtype)], axis=-1)


def _masked_argmax(logits: mx.array, token_mask: mx.array) -> mx.array:
    if token_mask.ndim == 1:
        token_mask = token_mask[None]
    rows = logits.shape[0] * logits.shape[1]
    output_size = logits.shape[-1]
    if (
        token_mask.dtype != mx.int32
        or token_mask.shape[0] != rows
        or token_mask.shape[1] < (output_size + 31) // 32
    ):
        raise ValueError(
            "packed token mask must be int32 with one complete row per token"
        )
    token_ids = mx.arange(output_size, dtype=mx.int32)
    words = token_mask[:, token_ids // 32]
    allowed = ((words >> (token_ids % 32)) & 1).astype(mx.bool_)
    flat_logits = logits.reshape(rows, output_size)
    masked = mx.where(allowed, flat_logits, -float("inf"))
    return mx.argmax(masked, axis=-1).reshape(logits.shape[:2])


def singleton_quantized_argmax(
    linear,
    x: mx.array,
    token_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Argmax over the batch-generic singleton-equivalent projection."""
    logits = singleton_quantized_linear(linear, x)
    if logits is None:
        return None
    if token_mask is not None:
        return _masked_argmax(logits, token_mask)
    return mx.argmax(logits, axis=-1)


def _stable_nvfp4_linear(linear, x: mx.array) -> Optional[mx.array]:
    if not isinstance(linear, nn.QuantizedLinear) or linear.mode != "nvfp4":
        return None
    return singleton_quantized_linear(linear, x)


def _stable_nvfp4_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    if not isinstance(linear, nn.QuantizedLinear) or linear.mode != "nvfp4":
        return None
    return singleton_quantized_argmax(linear, x, token_mask)


def _target_verify_qlinear_header(
    bits: int,
    group_size: int,
    results_per_simdgroup: int = 4,
    *,
    q3_shifted_fields: bool = False,
) -> str:
    # Shifted Q3 extraction is faster but changes floating-point operation
    # order. Use it only for argmax kernels, where exact token IDs are
    # verified; projection kernels retain the unshifted form for byte-identical
    # singleton outputs.
    return (
        r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr bool Q3_SHIFTED = __Q3_SHIFTED__;
    constant constexpr int PACK_FACTOR =
        ((BITS == 3 || BITS == 5) ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK =
        ((BITS == 3 || BITS == 5) ? (BITS == 3 ? 3 : 5) : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = __RESULTS_PER_SIMDGROUP__;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 3) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] +
              x[i + 5] + x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = Q3_SHIFTED ? x[i + 1] : x[i + 1] / 8.0f;
          x_thread[i + 2] = Q3_SHIFTED ? x[i + 2] : x[i + 2] / 64.0f;
          x_thread[i + 3] = Q3_SHIFTED ? x[i + 3] : x[i + 3] / 2.0f;
          x_thread[i + 4] = Q3_SHIFTED ? x[i + 4] : x[i + 4] / 16.0f;
          x_thread[i + 5] = Q3_SHIFTED ? x[i + 5] : x[i + 5] / 128.0f;
          x_thread[i + 6] = Q3_SHIFTED ? x[i + 6] : x[i + 6] / 4.0f;
          x_thread[i + 7] = Q3_SHIFTED ? x[i + 7] : x[i + 7] / 32.0f;
        }
      } else if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1];
          x_thread[i + 2] = x[i + 2];
          x_thread[i + 3] = x[i + 3];
        }
      } else if (BITS == 5) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
          sum += x[i];
          x_thread[i] = x[i];
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 3) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 3 * i;
          accum += (wb[0] & 0x07) * xt[0];
          accum +=
              (Q3_SHIFTED ? ((wb[0] >> 3) & 0x07) : (wb[0] & 0x38)) * xt[1];
          accum +=
              (Q3_SHIFTED ? (wb[0] >> 6) : (wb[0] & 0xc0)) * xt[2];
          accum +=
              (wb[1] & 0x01) * (xt[2] * (Q3_SHIFTED ? 4.0f : 256.0f));
          accum +=
              (Q3_SHIFTED ? ((wb[1] >> 1) & 0x07) : (wb[1] & 0x0e)) * xt[3];
          accum +=
              (Q3_SHIFTED ? ((wb[1] >> 4) & 0x07) : (wb[1] & 0x70)) * xt[4];
          accum +=
              (Q3_SHIFTED ? (wb[1] >> 7) : (wb[1] & 0x80)) * xt[5];
          accum +=
              (wb[2] & 0x03) * (xt[5] * (Q3_SHIFTED ? 2.0f : 256.0f));
          accum +=
              (Q3_SHIFTED ? ((wb[2] >> 2) & 0x07) : (wb[2] & 0x1c)) * xt[6];
          accum +=
              (Q3_SHIFTED ? (wb[2] >> 5) : (wb[2] & 0xe0)) * xt[7];
        }
      } else if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          uint packed = ws[i];
          accum +=
              (x_thread[4 * i] * (packed & 0x000f) +
               x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
               x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
               x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 5 * i;

          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
          accum += x_thread[i] * w[i];
        }
      }
      return scale * accum + sum * bias;
    }

    inline float qdot_exact(
        const thread uint16_t* ws,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 3) {
        const thread uint8_t* wb = (const thread uint8_t*)ws;
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const thread uint8_t* packed = wb + 3 * i;
          accum += (packed[0] & 0x07) * xt[0];
          accum +=
              (Q3_SHIFTED ? ((packed[0] >> 3) & 0x07) : (packed[0] & 0x38)) *
              xt[1];
          accum +=
              (Q3_SHIFTED ? (packed[0] >> 6) : (packed[0] & 0xc0)) * xt[2];
          accum +=
              (packed[1] & 0x01) * (xt[2] * (Q3_SHIFTED ? 4.0f : 256.0f));
          accum +=
              (Q3_SHIFTED ? ((packed[1] >> 1) & 0x07) : (packed[1] & 0x0e)) *
              xt[3];
          accum +=
              (Q3_SHIFTED ? ((packed[1] >> 4) & 0x07) : (packed[1] & 0x70)) *
              xt[4];
          accum +=
              (Q3_SHIFTED ? (packed[1] >> 7) : (packed[1] & 0x80)) * xt[5];
          accum +=
              (packed[2] & 0x03) * (xt[5] * (Q3_SHIFTED ? 2.0f : 256.0f));
          accum +=
              (Q3_SHIFTED ? ((packed[2] >> 2) & 0x07) : (packed[2] & 0x1c)) *
              xt[6];
          accum +=
              (Q3_SHIFTED ? (packed[2] >> 5) : (packed[2] & 0xe0)) * xt[7];
        }
      } else if (BITS == 4) {
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          uint packed = ws[i];
          accum +=
              (x_thread[4 * i] * (packed & 0x000f) +
               x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
               x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
               x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
        }
      }
      return scale * accum + sum * bias;
    }

""".replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__RESULTS_PER_SIMDGROUP__", str(results_per_simdgroup))
        .replace("__Q3_SHIFTED__", "true" if q3_shifted_fields else "false")
    )


_TARGET_VERIFY_QMV_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

#pragma clang loop unroll_count(2)
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
        }
      }
    }
"""


_TARGET_VERIFY_QARGMAX_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

#pragma clang loop unroll_count(2)
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N_SIZE) {
          float rounded = float(T(simd_sum(result[t][row])));
          if (rounded > best_value) {
            best_value = rounded;
            best_index = n;
          }
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset = (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }
"""


_TARGET_VERIFY_QMV_TOKEN_TILED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint token_tile = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int TOKEN_TILE = 2;
    constexpr int NUM_TOKEN_TILES = (VERIFY_T + TOKEN_TILE - 1) / TOKEN_TILE;
    int b_idx = int(token_tile) / NUM_TOKEN_TILES;
    int token_tile_idx = int(token_tile) - b_idx * NUM_TOKEN_TILES;
    int token_start = token_tile_idx * TOKEN_TILE;
    int token_count = min(TOKEN_TILE, VERIFY_T - token_start);
    int flat_token_start = b_idx * VERIFY_T + token_start;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + flat_token_start * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[TOKEN_TILE][RESULTS_PER_SIMDGROUP];
    float x_thread[TOKEN_TILE][VALUES_PER_THREAD];
    for (int t = 0; t < TOKEN_TILE; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[TOKEN_TILE];
      for (int t = 0; t < token_count; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < token_count; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < token_count; ++t) {
        float r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(flat_token_start + t) * N_SIZE + n] = T(r);
        }
      }
    }
"""


_TARGET_VERIFY_QARGMAX_TOKEN_TILED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint token_tile = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int TOKEN_TILE = 2;
    constexpr int NUM_TOKEN_TILES = (VERIFY_T + TOKEN_TILE - 1) / TOKEN_TILE;
    int b_idx = int(token_tile) / NUM_TOKEN_TILES;
    int token_tile_idx = int(token_tile) - b_idx * NUM_TOKEN_TILES;
    int token_start = token_tile_idx * TOKEN_TILE;
    int token_count = min(TOKEN_TILE, VERIFY_T - token_start);
    int flat_token_start = b_idx * VERIFY_T + token_start;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[TOKEN_TILE][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[TOKEN_TILE][NUM_SIMDGROUPS];

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + flat_token_start * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[TOKEN_TILE][RESULTS_PER_SIMDGROUP];
    float x_thread[TOKEN_TILE][VALUES_PER_THREAD];
    for (int t = 0; t < TOKEN_TILE; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[TOKEN_TILE];
      for (int t = 0; t < token_count; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < token_count; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < token_count; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N_SIZE) {
          float rounded = float(T(simd_sum(result[t][row])));
          if (rounded > best_value) {
            best_value = rounded;
            best_index = n;
          }
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < token_count; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset =
            (flat_token_start + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }
"""


_TARGET_VERIFY_QMV_STREAMED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint16_t* source =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        uint16_t packed[VALUES_PER_THREAD / 4];
        for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
          packed[index] = source[index];
        }
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        for (int t = 0; t < VERIFY_T; ++t) {
          float x_thread[VALUES_PER_THREAD];
          float sum = load_vector_exact<T>(
              xk + t * K_SIZE, x_thread);
          result[t][row] += qdot_exact(
              packed, x_thread, scale, bias, sum);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float value = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
        }
      }
    }
"""


_TARGET_VERIFY_QARGMAX_STREAMED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint16_t* source =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        uint16_t packed[VALUES_PER_THREAD / 4];
        for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
          packed[index] = source[index];
        }
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        for (int t = 0; t < VERIFY_T; ++t) {
          float x_thread[VALUES_PER_THREAD];
          float sum = load_vector_exact<T>(
              xk + t * K_SIZE, x_thread);
          result[t][row] += qdot_exact(
              packed, x_thread, scale, bias, sum);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        float rounded = float(T(simd_sum(result[t][row])));
        if (n < N_SIZE && rounded > best_value) {
          best_value = rounded;
          best_index = n;
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset =
            (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }
"""

_TARGET_VERIFY_MASKED_QARGMAX_SOURCE = _TARGET_VERIFY_QARGMAX_SOURCE.replace(
    "if (n < N_SIZE) {",
    """if (
          n < N_SIZE &&
          ((as_type<uint>(mask[
                (int(b_idx) * VERIFY_T + t) * mask_shape[1] + (n >> 5)]) >>
            (n & 31)) & 1u) != 0u) {""",
)

_TARGET_VERIFY_MASKED_QARGMAX_TOKEN_TILED_SOURCE = _TARGET_VERIFY_QARGMAX_TOKEN_TILED_SOURCE.replace(
    "if (n < N_SIZE) {",
    """if (
          n < N_SIZE &&
          ((as_type<uint>(mask[
                (flat_token_start + t) * mask_shape[1] + (n >> 5)]) >>
            (n & 31)) & 1u) != 0u) {""",
)

_TARGET_VERIFY_MASKED_QARGMAX_STREAMED_SOURCE = _TARGET_VERIFY_QARGMAX_STREAMED_SOURCE.replace(
    "if (n < N_SIZE && rounded > best_value) {",
    """if (
          n < N_SIZE &&
          ((as_type<uint>(mask[
                (int(b_idx) * VERIFY_T + t) * mask_shape[1] + (n >> 5)]) >>
            (n & 31)) & 1u) != 0u &&
          rounded > best_value) {""",
)


def _target_verify_fused_qmv_source(source: str, n_sizes) -> str:
    selections = []
    offset = int(n_sizes[0])
    for index, n_size in enumerate(n_sizes[1:], start=1):
        selections.append(f"""    if (global_out >= {offset}) {{
      local_out = global_out - {offset};
      selected_w = (const device uint8_t*)w{index};
      selected_scales = scales{index};
      selected_biases = biases{index};
    }}""")
        offset += int(n_size)
    selection = "\n".join(selections)
    source = source.replace(
        "    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;",
        f"""    int global_out = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int local_out = global_out;
    const device uint8_t* selected_w = (const device uint8_t*)w0;
    const device T* selected_scales = scales0;
    const device T* selected_biases = biases0;
{selection}""",
    )
    return (
        source.replace(
            "(const device uint8_t*)w + out_row * in_vec_size_w",
            "selected_w + local_out * in_vec_size_w",
        )
        .replace(
            "scales + out_row * in_vec_size_g",
            "selected_scales + local_out * in_vec_size_g",
        )
        .replace(
            "biases + out_row * in_vec_size_g",
            "selected_biases + local_out * in_vec_size_g",
        )
        .replace("int n = out_row + row;", "int n = global_out + row;")
    )


def _projection_results_per_simdgroup(bits, verify_t):
    # Q3's unshifted projection arithmetic benefits from a smaller output
    # tile at T=3/4. Keep the singleton reduction order within each output.
    return 2 if bits == 3 and verify_t in (3, 4) else 4


@lru_cache(maxsize=None)
def _target_verify_qmv_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_target_verify_qlinear_header(
            bits, group_size, _projection_results_per_simdgroup(bits, verify_t)
        ),
        source=_TARGET_VERIFY_QMV_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qargmax_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qargmax_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(
            bits, group_size, q3_shifted_fields=bits == 3
        ),
        source=_TARGET_VERIFY_QARGMAX_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_masked_qargmax_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_masked_qargmax_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases", "mask"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(
            bits, group_size, q3_shifted_fields=bits == 3
        ),
        source=_TARGET_VERIFY_MASKED_QARGMAX_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qmv_token_tiled_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qmv_token_tiled_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QMV_TOKEN_TILED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qargmax_token_tiled_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qargmax_token_tiled_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QARGMAX_TOKEN_TILED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_masked_qargmax_token_tiled_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_masked_qargmax_token_tiled_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases", "mask"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_MASKED_QARGMAX_TOKEN_TILED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qmv_streamed_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qmv_streamed_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size, 1),
        source=_TARGET_VERIFY_QMV_STREAMED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qargmax_streamed_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_qargmax_streamed_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size, 1),
        source=_TARGET_VERIFY_QARGMAX_STREAMED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_masked_qargmax_streamed_kernel(
    bits, group_size, dtype, verify_t, k_size, n_size
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_masked_qargmax_streamed_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases", "mask"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size, 1),
        source=_TARGET_VERIFY_MASKED_QARGMAX_STREAMED_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_fused_qmv_kernel(bits, group_size, dtype, verify_t, k_size, n_sizes):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    shape_name = "_".join(str(n) for n in n_sizes)
    input_names = ["x"]
    for index in range(len(n_sizes)):
        input_names.extend([f"w{index}", f"scales{index}", f"biases{index}"])
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_fused_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{shape_name}_{dtype_name}"
        ),
        input_names=input_names,
        output_names=["y"],
        header=_target_verify_qlinear_header(
            bits, group_size, _projection_results_per_simdgroup(bits, verify_t)
        ),
        source=_target_verify_fused_qmv_source(_TARGET_VERIFY_QMV_SOURCE, n_sizes),
    )


@lru_cache(maxsize=None)
def _target_verify_fused_qmv_streamed_kernel(
    bits, group_size, dtype, verify_t, k_size, n_sizes
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    shape_name = "_".join(str(n) for n in n_sizes)
    input_names = ["x"]
    for index in range(len(n_sizes)):
        input_names.extend([f"w{index}", f"scales{index}", f"biases{index}"])
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_affine_fused_qmv_streamed_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{shape_name}_{dtype_name}"
        ),
        input_names=input_names,
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size, 1),
        source=_target_verify_fused_qmv_source(
            _TARGET_VERIFY_QMV_STREAMED_SOURCE, n_sizes
        ),
    )


def supports_optimized_affine_head(linear) -> bool:
    """Return whether the exact affine verifier kernel supports ``linear``."""
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or linear.bits not in (3, 4, 5, 8)
        or linear.mode != "affine"
        or linear.biases is None
        or linear.scales.dtype not in (mx.bfloat16, mx.float16)
        or linear.biases.dtype != linear.scales.dtype
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    N = linear.weight.shape[0]
    return K % 512 == 0 and N % 8 == 0


def _can_optimized_affine_linear(linear, x: mx.array) -> bool:
    """Check the exact affine kernel for a runtime-sized batch."""
    if (
        not supports_optimized_affine_head(linear)
        or x.ndim != 3
        or x.shape[1] < 1
        or x.dtype != linear.scales.dtype
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    return x.shape[-1] == K


def optimized_affine_linear(linear, x: mx.array) -> Optional[mx.array]:
    if not _can_optimized_affine_linear(linear, x):
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]

    x = mx.contiguous(x)
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    results_per_simdgroup = (
        1 if streamed else _projection_results_per_simdgroup(linear.bits, T)
    )
    if streamed:
        kernel_factory = _target_verify_qmv_streamed_kernel
    elif token_tiled:
        kernel_factory = _target_verify_qmv_token_tiled_kernel
    else:
        kernel_factory = _target_verify_qmv_kernel
    kernel_args = (linear.bits, linear.group_size, x.dtype, T, K, N)
    kernel = kernel_factory(*kernel_args)
    rows_per_threadgroup = 2 * results_per_simdgroup
    out = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(
            32,
            2 * (N // rows_per_threadgroup),
            B * ((T + 1) // 2) if token_tiled else B,
        ),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def optimized_affine_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    if not _can_optimized_affine_linear(linear, x) or "bias" in linear:
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    results_per_simdgroup = 1 if streamed else 4
    rows_per_threadgroup = 2 * results_per_simdgroup
    num_tiles = N // rows_per_threadgroup

    x = mx.contiguous(x)
    if streamed:
        kernel_factory = (
            _target_verify_masked_qargmax_streamed_kernel
            if token_mask is not None
            else _target_verify_qargmax_streamed_kernel
        )
    elif token_tiled:
        kernel_factory = (
            _target_verify_masked_qargmax_token_tiled_kernel
            if token_mask is not None
            else _target_verify_qargmax_token_tiled_kernel
        )
    else:
        kernel_factory = (
            _target_verify_masked_qargmax_kernel
            if token_mask is not None
            else _target_verify_qargmax_kernel
        )
    kernel_args = (linear.bits, linear.group_size, x.dtype, T, K, N)
    kernel = kernel_factory(*kernel_args)
    inputs = [x, linear.weight, linear.scales, linear.biases]
    if token_mask is not None:
        if token_mask.ndim == 1:
            token_mask = token_mask[None, :]
        if (
            token_mask.dtype != mx.int32
            or token_mask.shape[0] != B * T
            or token_mask.shape[1] < (N + 31) // 32
        ):
            raise ValueError(
                "packed token mask must be int32 with one complete row per token"
            )
        inputs.append(token_mask)
    tile_values, tile_indices = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
            ("NUM_TILES", int(num_tiles)),
        ],
        grid=(
            32,
            2 * num_tiles,
            B * ((T + 1) // 2) if token_tiled else B,
        ),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, num_tiles), (B, T, num_tiles)],
        output_dtypes=[x.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(tile_indices, best_tile[..., None], axis=-1).squeeze(-1)


def optimized_affine_linears(linears, x: mx.array):
    bits = getattr(linears[0], "bits", None) if linears else None
    if (
        not 2 <= len(linears) <= 4
        or x.ndim != 3
        or not 1 < x.shape[1] <= 8
        or bits not in (3, 4, 5, 8)
        or not all(
            isinstance(linear, nn.QuantizedLinear)
            and linear.bits == bits
            and linear.group_size == linears[0].group_size
            and linear.mode == linears[0].mode
            and "bias" not in linear
            and _can_optimized_affine_linear(linear, x)
            for linear in linears
        )
    ):
        return None

    B, T, K = x.shape
    n_sizes = tuple(int(linear.weight.shape[0]) for linear in linears)
    total_n = sum(n_sizes)
    x = mx.contiguous(x)
    streamed = bits == 4 and T >= 6
    kernel_factory = (
        _target_verify_fused_qmv_streamed_kernel
        if streamed
        else _target_verify_fused_qmv_kernel
    )
    kernel = kernel_factory(bits, linears[0].group_size, x.dtype, T, K, n_sizes)
    inputs = [x]
    for linear in linears:
        inputs.extend([linear.weight, linear.scales, linear.biases])
    rows_per_threadgroup = (
        2 if streamed else 2 * _projection_results_per_simdgroup(bits, T)
    )
    out = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(total_n)),
        ],
        grid=(32, 2 * (total_n // rows_per_threadgroup), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, total_n)],
        output_dtypes=[x.dtype],
    )[0]
    split_indices = []
    offset = 0
    for n_size in n_sizes[:-1]:
        offset += n_size
        split_indices.append(offset)
    return tuple(mx.split(out, split_indices, axis=-1))


_NVFP4_ARGMAX_HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int VALUES_PER_THREAD = 8;
    constant constexpr int BLOCK_SIZE = SIMD_SIZE * VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int ROWS_PER_THREADGROUP =
        RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    inline float decode_e2m1(uint value) {
        half decoded = as_type<half>(ushort((value & 7) << 9));
        decoded *= 16384.0h;
        return float(value & 8 ? -decoded : decoded);
    }

    inline float decode_e4m3(uint8_t value) {
        half decoded = as_type<half>(ushort((value & 127) << 7));
        decoded *= 256.0h;
        return float(value & 128 ? -decoded : decoded);
    }

    inline float nvfp4_dot(
        const device uint8_t* weight,
        const thread float* values,
        float scale) {
        const device uint16_t* packed =
            reinterpret_cast<const device uint16_t*>(weight);
        float result = 0.0f;
        for (int index = 0; index < 2; ++index) {
            uint word = packed[index];
            result +=
                values[4 * index] * decode_e2m1(word) +
                values[4 * index + 1] * decode_e2m1(word >> 4) +
                values[4 * index + 2] * decode_e2m1(word >> 8) +
                values[4 * index + 3] * decode_e2m1(word >> 12);
        }
        return scale * result;
    }
"""

_NVFP4_ARGMAX_SOURCE = r"""
    uint output_tile = threadgroup_position_in_grid.y;
    uint batch_index = threadgroup_position_in_grid.z;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint output_row =
        output_tile * ROWS_PER_THREADGROUP +
        simd_group * RESULTS_PER_SIMDGROUP;

    threadgroup float group_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int group_indices[VERIFY_T][NUM_SIMDGROUPS];
    float results[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float values[VERIFY_T][VALUES_PER_THREAD];
    for (int token = 0; token < VERIFY_T; ++token) {
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            results[token][row] = 0.0f;
        }
    }

    const device uint8_t* weight_bytes =
        reinterpret_cast<const device uint8_t*>(w);
    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
        int lane_start = k + int(lane) * VALUES_PER_THREAD;
        if (lane_start < K_SIZE) {
            for (int token = 0; token < VERIFY_T; ++token) {
                const device T* input =
                    x + (int(batch_index) * VERIFY_T + token) * K_SIZE + lane_start;
                for (int index = 0; index < VALUES_PER_THREAD; ++index) {
                    values[token][index] = float(input[index]);
                }
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                int n = int(output_row) + row;
                const device uint8_t* weight =
                    weight_bytes + n * (K_SIZE / 2) + lane_start / 2;
                float scale = decode_e4m3(
                    scales[n * (K_SIZE / 16) + lane_start / 16]
                );
                for (int token = 0; token < VERIFY_T; ++token) {
                    results[token][row] += nvfp4_dot(weight, values[token], scale);
                }
            }
        }
    }

    for (int token = 0; token < VERIFY_T; ++token) {
        float best_value = -3.4028234663852886e38f;
        int best_index = 0;
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = int(output_row) + row;
            float value = float(T(simd_sum(results[token][row])));
            if (value > best_value) {
                best_value = value;
                best_index = n;
            }
        }
        if (lane == 0) {
            group_values[token][simd_group] = best_value;
            group_indices[token][simd_group] = best_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && lane == 0) {
        for (int token = 0; token < VERIFY_T; ++token) {
            float best_value = group_values[token][0];
            int best_index = group_indices[token][0];
            for (int group = 1; group < NUM_SIMDGROUPS; ++group) {
                float value = group_values[token][group];
                if (value > best_value) {
                    best_value = value;
                    best_index = group_indices[token][group];
                }
            }
            int offset =
                (int(batch_index) * VERIFY_T + token) * NUM_TILES +
                int(output_tile);
            tile_values[offset] = T(best_value);
            tile_indices[offset] = best_index;
        }
    }
"""


@lru_cache(maxsize=None)
def _nvfp4_argmax_kernel(dtype, verify_length, input_size, output_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_nvfp4_argmax_"
            f"t{verify_length}_k{input_size}_n{output_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["tile_values", "tile_indices"],
        header=_NVFP4_ARGMAX_HEADER,
        source=_NVFP4_ARGMAX_SOURCE,
    )


def optimized_nvfp4_argmax(
    linear,
    hidden: mx.array,
    token_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    if token_mask is not None:
        return None
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or linear.mode != "nvfp4"
        or linear.bits != 4
        or linear.group_size != 16
        or "bias" in linear
        or hidden.ndim != 3
        or hidden.dtype not in (mx.bfloat16, mx.float16)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None

    batch, length, input_size = hidden.shape
    expected_input_size = linear.weight.shape[1] * 8
    output_size = linear.weight.shape[0]
    if input_size != expected_input_size or input_size % 16 or output_size % 8:
        return None

    tile_count = output_size // 8
    kernel = _nvfp4_argmax_kernel(hidden.dtype, length, input_size, output_size)
    tile_values, tile_indices = kernel(
        inputs=[mx.contiguous(hidden), linear.weight, linear.scales],
        template=[
            ("T", hidden.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(input_size)),
            ("N_SIZE", int(output_size)),
            ("NUM_TILES", int(tile_count)),
        ],
        grid=(32, 2 * tile_count, batch),
        threadgroup=(32, 2, 1),
        output_shapes=[
            (batch, length, tile_count),
            (batch, length, tile_count),
        ],
        output_dtypes=[hidden.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(tile_indices, best_tile[..., None], axis=-1).squeeze(-1)


def decode_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Decode projection: singleton arithmetic for NVFP4, native batch otherwise."""
    if not supports_quantization(linear):
        return None
    output = _stable_nvfp4_linear(linear, x)
    return output if output is not None else exact_quantized_linear(linear, x)


def decode_quantized_argmax(
    linear,
    x: mx.array,
    token_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Argmax using the same numerical contract as decode_quantized_linear."""
    if not supports_quantization(linear):
        return None
    output = _stable_nvfp4_argmax(linear, x, token_mask)
    if output is not None:
        return output
    logits = decode_quantized_linear(linear, x)
    if logits is None:
        return None
    return (
        _masked_argmax(logits, token_mask)
        if token_mask is not None
        else mx.argmax(logits, axis=-1)
    )


__all__ = [
    "AFFINE_BITS",
    "FIXED_FORMATS",
    "decode_quantized_linear",
    "decode_quantized_argmax",
    "exact_quantized_linear",
    "exact_quantized_moe_hc_expand",
    "exact_quantized_selected_linear",
    "exact_quantized_switch_linear",
    "optimized_affine_argmax",
    "optimized_affine_linear",
    "optimized_affine_linears",
    "optimized_nvfp4_argmax",
    "pad_token_mask",
    "singleton_quantized_argmax",
    "singleton_quantized_linear",
    "supports_optimized_affine_head",
    "supports_quantization",
]
