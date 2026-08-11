from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_EXACT_AFFINE_QMV_HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
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
      if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
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
      }
      return scale * accum + sum * bias;
    }
"""


_EXACT_AFFINE_QMV_SOURCE = r"""
    uint token_idx = threadgroup_position_in_grid.x;
    uint n_tile = threadgroup_position_in_grid.y;
    uint batch_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scale_ptr =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bias_ptr =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_ptr =
        x + (int(batch_idx) * VERIFY_T + int(token_idx)) * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float result[RESULTS_PER_SIMDGROUP] = {0};
    float x_thread[VALUES_PER_THREAD];

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector_exact<T>(x_ptr, x_thread);
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* weight_ptr = ws + row * in_vec_size_w;
        const device T* row_scale = scale_ptr + row * in_vec_size_g;
        const device T* row_bias = bias_ptr + row * in_vec_size_g;
        result[row] += qdot_exact(
            weight_ptr,
            x_thread,
            float(row_scale[0]),
            float(row_bias[0]),
            sum);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      scale_ptr += BLOCK_SIZE / GS;
      bias_ptr += BLOCK_SIZE / GS;
      x_ptr += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[(int(batch_idx) * VERIFY_T + int(token_idx)) * N_SIZE + out_row + row] =
            T(result[row]);
      }
    }
"""


@lru_cache(maxsize=None)
def _exact_affine_qmv_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "mlx_vlm_exact_affine_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_EXACT_AFFINE_QMV_HEADER.replace("__BITS__", str(bits)).replace(
            "__GS__", str(group_size)
        ),
        source=_EXACT_AFFINE_QMV_SOURCE,
    )


def exact_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Run affine QMV independently per token with singleton-decode rounding."""

    if (
        not mx.metal.is_available()
        or not isinstance(linear, nn.QuantizedLinear)
        or linear.bits not in (4, 5)
        or linear.mode != "affine"
        or linear.biases is None
        or x.ndim != 3
        or x.shape[1] <= 1
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return None

    batch, length, hidden_size = x.shape
    output_size = linear.weight.shape[0]
    packed_hidden_size = linear.weight.shape[1] * 32 // linear.bits
    if (
        hidden_size != packed_hidden_size
        or hidden_size % 512
        or output_size % 8
        or linear.group_size % 16
    ):
        return None

    x = mx.contiguous(x)
    kernel = _exact_affine_qmv_kernel(
        linear.bits,
        linear.group_size,
        x.dtype,
        length,
        hidden_size,
        output_size,
    )
    output = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(hidden_size)),
            ("N_SIZE", int(output_size)),
        ],
        grid=(32 * length, 2 * (output_size // 8), batch),
        threadgroup=(32, 2, 1),
        output_shapes=[(batch, length, output_size)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        output = output + linear["bias"]
    return output


__all__ = ["exact_quantized_linear"]
