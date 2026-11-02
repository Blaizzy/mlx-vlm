"""Batched verifier projections with the native small-batch reduction order."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _wide_kernel():
    return mx.fast.metal_kernel(
        name="ml4_verify_wide_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        header=_HEADER,
        source=r"""
        GemvWide<T, 4, KL, float>::run(
            weight, x, weight, out,
            x_shape[2], weight_shape[0], x_shape[0]*x_shape[1],
            x_shape[2], x_shape[2], 1.0f, 0.0f, 0, 0,
            threadgroup_position_in_grid, threadgroups_per_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """,
    )


def native_wide_linear(weight, x):
    """Reuse dense weights across positions without changing K-lane reductions."""
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
    architecture = mx.device_info().get("architecture", "")
    # The native wide-GEMV dispatch starts with Apple GPU generation 15 (M3).
    generation = architecture.split("_g")[-1][:2]
    if not generation.isdigit() or int(generation) < 15:
        return None
    passes = (x.shape[0] + 4) // 5
    lanes = 32 if passes == 1 or weight.shape[0] <= 64 else 16
    groups = lanes // 8
    return _wide_kernel()(
        inputs=[x, weight],
        template=[("T", x.dtype), ("KL", lanes)],
        grid=(32, ((weight.shape[0] + 3) // 4) * groups, 1),
        threadgroup=(32, groups, 1),
        output_shapes=[(*x.shape[:-1], weight.shape[0])],
        output_dtypes=[x.dtype],
    )[0]


_HEADER = r"""

/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copied from MLX 1ba7f969ad05d86decfeeb83787cc9b79e440962:
mlx/backend/metal/kernels/gemv.h.
*/

// GemvWide from MLX 1ba7f969, mlx/backend/metal/kernels/gemv.h.
#define MLX_MTL_PRAGMA_UNROLL _Pragma("unroll")
constant constexpr bool gemv_wide_do_axpby = false;
template <
    typename T,
    const int vecs_per_tg,
    const int k_lanes,
    typename AccT = float>
struct GemvWide {
  static constexpr constant int unroll = 8;
  static constexpr constant int num_simdgroups = k_lanes / 8;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      int in_vec_size,
      int out_vec_size,
      int M,
      int matrix_ld,
      int vector_ld,
      float alpha,
      float beta,
      int bias_ld,
      int bias_fd,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 tgpg [[threadgroups_per_grid]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int rows_per_simdgroup = 32 / k_lanes;
    constexpr int rows_per_tg = rows_per_simdgroup * num_simdgroups;

    const short k_lane =
        simd_lid % k_lanes; // this lane's slot in the K reduction
    const short sg_row =
        simd_lid / k_lanes; // which output row of the simdgroup

    const int out_row =
        tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;

    // Clamped tail rows/vectors read valid memory; the guarded writes
    // below never store them.
    const int row = min(out_row, out_vec_size - 1);
    const device T* wrow = mat + int64_t(row) * matrix_ld;
    const device vec<T, 4>* w4 = (const device vec<T, 4>*)wrow;
    const int n_v4 = in_vec_size / 4;
    const int n_main = n_v4 - n_v4 % (k_lanes * unroll);

    // Vector chunks beyond grid.x round-robin onto the same threadgroups:
    // re-walking the row block hits cache where an extra grid column would
    // re-stream it from DRAM.
    const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;
    const int x_ld4 = vector_ld / 4;
    const int n_chunks = (M + vecs_per_tg - 1) / vecs_per_tg;
    for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
      const int vec0 = chunk * vecs_per_tg;

      int x_off4[vecs_per_tg];
      for (int v = 0; v < vecs_per_tg; v++) {
        x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
      }

      AccT result[vecs_per_tg] = {0};

      // Adjacent lanes read adjacent blocks so every transaction lands on
      // whole cachelines; the unroll keeps loads in flight on short rows.
      for (int base = 0; base < n_main; base += k_lanes * unroll) {
        vec<AccT, 4> wf[unroll];
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < unroll; i++) {
          wf[i] = vec<AccT, 4>(w4[base + i * k_lanes + k_lane]);
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int v = 0; v < vecs_per_tg; v++) {
          AccT acc = 0;
          MLX_MTL_PRAGMA_UNROLL
          for (int i = 0; i < unroll; i++) {
            acc +=
                dot(wf[i],
                    vec<AccT, 4>(x4[x_off4[v] + base + i * k_lanes + k_lane]));
          }
          result[v] += acc;
        }
      }
      for (int idx = n_main + k_lane; idx < n_v4; idx += k_lanes) {
        const vec<AccT, 4> wf = vec<AccT, 4>(w4[idx]);
        MLX_MTL_PRAGMA_UNROLL
        for (int v = 0; v < vecs_per_tg; v++) {
          result[v] += dot(wf, vec<AccT, 4>(x4[x_off4[v] + idx]));
        }
      }

      // The halving shuffles reduce each row's k_lanes while rows sharing
      // the simdgroup stay separate.
      MLX_MTL_PRAGMA_UNROLL
      for (int v = 0; v < vecs_per_tg; v++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort off = k_lanes / 2; off >= 1; off >>= 1) {
          result[v] += simd_shuffle_down(result[v], off);
        }
      }

      if (k_lane == 0 && out_row < out_vec_size) {
        for (int v = 0; v < vecs_per_tg; v++) {
          if (vec0 + v < M) {
            int64_t out_idx = int64_t(vec0 + v) * out_vec_size + out_row;
            if (gemv_wide_do_axpby) {
              AccT bias_val = static_cast<AccT>(
                  bias[int64_t(vec0 + v) * bias_ld + out_row * bias_fd]);
              out_vec[out_idx] =
                  static_cast<T>(alpha * result[v] + beta * bias_val);
            } else {
              out_vec[out_idx] = static_cast<T>(result[v]);
            }
          }
        }
      }
    }
  }
};

"""
