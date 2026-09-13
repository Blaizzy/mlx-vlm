"""Optimized exact MXFP4 operations for speculative verification."""

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# MLX's wide MXFP4 matmul uses a different reduction order from singleton
# ``QuantizedLinear`` calls.  Keep that singleton QMV order for each output row
# while reusing the loaded inputs across the short speculative time dimension.
_TARGET_VERIFY_MXFP4_HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int VERIFY_RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int VERIFY_NUM_SIMDGROUPS = 2;
    constant constexpr int VERIFY_ROWS_PER_THREADGROUP =
        VERIFY_RESULTS_PER_SIMDGROUP * VERIFY_NUM_SIMDGROUPS;
    constant constexpr int MXFP4_GROUP_SIZE = 32;
    constant constexpr int MXFP4_PACK_FACTOR = 8;
    constant constexpr int MXFP4_BYTES_PER_PACK = 4;
    constant constexpr int MXFP4_PACKS_PER_THREAD = 2;
    constant constexpr int MXFP4_VALUES_PER_THREAD =
        MXFP4_PACK_FACTOR * MXFP4_PACKS_PER_THREAD;
    constant constexpr int MXFP4_BLOCK_SIZE =
        MXFP4_VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int MXFP4_SCALE_STEP_PER_THREAD =
        MXFP4_GROUP_SIZE / MXFP4_VALUES_PER_THREAD;

    inline float decode_mxfp4_value(uint code) {
      half converted = as_type<half>(ushort((code & 7) << 9));
      converted *= 16384.0;
      float value = float(converted);
      return code & 8 ? -value : value;
    }

    inline float decode_mxfp4_scale(uint8_t encoded) {
      uint bits = encoded == 0 ? 0x00400000u : uint(encoded) << 23;
      return as_type<float>(bits);
    }

    template <typename T>
    inline void load_mxfp4_vector_exact(
        const device T* x,
        thread float* x_thread) {
      for (int i = 0; i < MXFP4_VALUES_PER_THREAD; ++i) {
        x_thread[i] = float(x[i]);
      }
    }

    inline float mxfp4_qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale) {
      float accum = 0.0f;
      const device uint16_t* ws = (const device uint16_t*)w;
      for (int i = 0; i < (MXFP4_VALUES_PER_THREAD / 4); ++i) {
        uint packed = ws[i];
        accum +=
            (x_thread[4 * i] * decode_mxfp4_value(packed) +
             x_thread[4 * i + 1] * decode_mxfp4_value(packed >> 4) +
             x_thread[4 * i + 2] * decode_mxfp4_value(packed >> 8) +
             x_thread[4 * i + 3] * decode_mxfp4_value(packed >> 12));
      }
      return scale * accum;
    }
"""


_TARGET_VERIFY_MXFP4_QMV_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row =
        int(n_tile) * VERIFY_ROWS_PER_THREADGROUP +
        int(simd_gid) * VERIFY_RESULTS_PER_SIMDGROUP;
    int in_vec_size_w =
        K_SIZE * MXFP4_BYTES_PER_PACK / MXFP4_PACK_FACTOR;
    int in_vec_size_g = K_SIZE / MXFP4_GROUP_SIZE;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * MXFP4_PACKS_PER_THREAD * MXFP4_BYTES_PER_PACK;
    const device uint8_t* sc =
        scales + out_row * in_vec_size_g +
        int(simd_lid) / MXFP4_SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE +
        int(simd_lid) * MXFP4_VALUES_PER_THREAD;

    float result[VERIFY_T][VERIFY_RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][MXFP4_VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < VERIFY_RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += MXFP4_BLOCK_SIZE) {
      for (int t = 0; t < VERIFY_T; ++t) {
        load_mxfp4_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

      for (int row = 0; row < VERIFY_RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device uint8_t* sl = sc + row * in_vec_size_g;
        float scale = decode_mxfp4_scale(sl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += mxfp4_qdot_exact(wl, x_thread[t], scale);
        }
      }

      ws +=
          MXFP4_BLOCK_SIZE * MXFP4_BYTES_PER_PACK / MXFP4_PACK_FACTOR;
      sc += MXFP4_BLOCK_SIZE / MXFP4_GROUP_SIZE;
      xk += MXFP4_BLOCK_SIZE;
    }

    for (int row = 0; row < VERIFY_RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float reduced = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(reduced);
        }
      }
    }
"""


_TARGET_VERIFY_MXFP4_QARGMAX_SOURCE = _TARGET_VERIFY_MXFP4_QMV_SOURCE.replace(
    """    for (int row = 0; row < VERIFY_RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float reduced = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(reduced);
        }
      }
    }""",
    """    threadgroup float tile_best_values[VERIFY_T][VERIFY_NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][VERIFY_NUM_SIMDGROUPS];

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < VERIFY_RESULTS_PER_SIMDGROUP; ++row) {
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
        for (int i = 1; i < VERIFY_NUM_SIMDGROUPS; ++i) {
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
    }""",
)

_TARGET_VERIFY_MASKED_MXFP4_QARGMAX_SOURCE = _TARGET_VERIFY_MXFP4_QARGMAX_SOURCE.replace(
    "if (n < N_SIZE && rounded > best_value) {",
    """if (
          n < N_SIZE &&
          ((as_type<uint>(mask[
                (int(b_idx) * VERIFY_T + t) * mask_shape[1] + (n >> 5)]) >>
            (n & 31)) & 1u) != 0u &&
          rounded > best_value) {""",
)


def _optimized_fused_mxfp4_qmv_source(n_sizes) -> str:
    selections = []
    offset = int(n_sizes[0])
    for index, n_size in enumerate(n_sizes[1:], start=1):
        selections.append(f"""    if (global_out >= {offset}) {{
      local_out = global_out - {offset};
      selected_w = (const device uint8_t*)w{index};
      selected_scales = scales{index};
    }}""")
        offset += int(n_size)
    selection = "\n".join(selections)
    source = _TARGET_VERIFY_MXFP4_QMV_SOURCE.replace(
        "    int in_vec_size_w =\n        K_SIZE * MXFP4_BYTES_PER_PACK / MXFP4_PACK_FACTOR;",
        f"""    int global_out = out_row;
    int local_out = global_out;
    const device uint8_t* selected_w = (const device uint8_t*)w0;
    const device uint8_t* selected_scales = scales0;
{selection}
    int in_vec_size_w =
        K_SIZE * MXFP4_BYTES_PER_PACK / MXFP4_PACK_FACTOR;""",
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
        .replace("int n = out_row + row;", "int n = global_out + row;")
    )


@lru_cache(maxsize=None)
def _optimized_mxfp4_qmv_kernel(dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_mxfp4_qmv_"
            f"t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["y"],
        header=_TARGET_VERIFY_MXFP4_HEADER,
        source=_TARGET_VERIFY_MXFP4_QMV_SOURCE,
    )


@lru_cache(maxsize=None)
def _optimized_mxfp4_qargmax_kernel(dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_mxfp4_qargmax_"
            f"t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["tile_values", "tile_indices"],
        header=_TARGET_VERIFY_MXFP4_HEADER,
        source=_TARGET_VERIFY_MXFP4_QARGMAX_SOURCE,
    )


@lru_cache(maxsize=None)
def _optimized_masked_mxfp4_qargmax_kernel(dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_masked_mxfp4_qargmax_"
            f"t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "mask"],
        output_names=["tile_values", "tile_indices"],
        header=_TARGET_VERIFY_MXFP4_HEADER,
        source=_TARGET_VERIFY_MASKED_MXFP4_QARGMAX_SOURCE,
    )


@lru_cache(maxsize=None)
def _optimized_fused_mxfp4_qmv_kernel(dtype, verify_t, k_size, n_sizes):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    shape_name = "_".join(str(n) for n in n_sizes)
    input_names = ["x"]
    for index in range(len(n_sizes)):
        input_names.extend([f"w{index}", f"scales{index}"])
    return mx.fast.metal_kernel(
        name=(
            "quantized_verify_fused_mxfp4_qmv_"
            f"t{verify_t}_k{k_size}_n{shape_name}_{dtype_name}"
        ),
        input_names=input_names,
        output_names=["y"],
        header=_TARGET_VERIFY_MXFP4_HEADER,
        source=_optimized_fused_mxfp4_qmv_source(n_sizes),
    )


def supports_optimized_mxfp4_head(linear) -> bool:
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or linear.mode != "mxfp4"
        or linear.bits != 4
        or linear.group_size != 32
        or linear.biases is not None
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    N = linear.weight.shape[0]
    return K % 512 == 0 and N % 8 == 0


def _can_optimized_mxfp4(linear, x: mx.array) -> bool:
    return (
        supports_optimized_mxfp4_head(linear)
        and x.ndim == 3
        and 1 <= x.shape[1] <= 4
        and x.dtype in (mx.bfloat16, mx.float16)
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.shape[-1] == linear.weight.shape[1] * 32 // linear.bits
    )


def optimized_mxfp4_linear(linear, x: mx.array) -> Optional[mx.array]:
    if not _can_optimized_mxfp4(linear, x):
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]
    x = mx.contiguous(x)
    kernel = _optimized_mxfp4_qmv_kernel(x.dtype, T, K, N)
    out = kernel(
        inputs=[x, linear.weight, linear.scales],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // 8), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def optimized_mxfp4_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    if not _can_optimized_mxfp4(linear, x) or "bias" in linear:
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]
    num_tiles = N // 8
    x = mx.contiguous(x)
    kernel_factory = (
        _optimized_masked_mxfp4_qargmax_kernel
        if token_mask is not None
        else _optimized_mxfp4_qargmax_kernel
    )
    kernel = kernel_factory(x.dtype, T, K, N)
    inputs = [x, linear.weight, linear.scales]
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
        grid=(32, 2 * num_tiles, B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, num_tiles), (B, T, num_tiles)],
        output_dtypes=[x.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(tile_indices, best_tile[..., None], axis=-1).squeeze(-1)


def optimized_mxfp4_linears(linears, x: mx.array):
    if (
        not 2 <= len(linears) <= 4
        or x.ndim != 3
        or not 1 < x.shape[1] <= 4
        or not all(
            "bias" not in linear and _can_optimized_mxfp4(linear, x)
            for linear in linears
        )
    ):
        return None

    B, T, K = x.shape
    n_sizes = tuple(int(linear.weight.shape[0]) for linear in linears)
    total_n = sum(n_sizes)
    x = mx.contiguous(x)
    kernel = _optimized_fused_mxfp4_qmv_kernel(x.dtype, T, K, n_sizes)
    inputs = [x]
    for linear in linears:
        inputs.extend([linear.weight, linear.scales])
    out = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(total_n)),
        ],
        grid=(32, 2 * (total_n // 8), B),
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


__all__ = [
    "optimized_mxfp4_argmax",
    "optimized_mxfp4_linear",
    "optimized_mxfp4_linears",
    "supports_optimized_mxfp4_head",
]
