from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    kv_sequence_length,
    scaled_dot_product_attention,
    slice_kv_sequence,
)
from ..exact_speculative_verify import exact_speculative_verify_dense_available
from ..exact_speculative_verify import (
    exact_speculative_verify_weight as _target_verify_weight,
)
from .gated_delta import gated_delta_update_with_states


def _use_target_verify_dense(linear, x: mx.array) -> bool:
    return (
        exact_speculative_verify_dense_available()
        and x.ndim == 3
        and x.shape[1] > 1
        and isinstance(linear, (nn.Linear, nn.QuantizedLinear))
    )


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
          accum += (Q3_SHIFTED ? ((wb[0] >> 3) & 0x07) : (wb[0] & 0x38)) * xt[1];
          accum += (Q3_SHIFTED ? (wb[0] >> 6) : (wb[0] & 0xc0)) * xt[2];
          accum += (wb[1] & 0x01) * (xt[2] * (Q3_SHIFTED ? 4.0f : 256.0f));
          accum += (Q3_SHIFTED ? ((wb[1] >> 1) & 0x07) : (wb[1] & 0x0e)) * xt[3];
          accum += (Q3_SHIFTED ? ((wb[1] >> 4) & 0x07) : (wb[1] & 0x70)) * xt[4];
          accum += (Q3_SHIFTED ? (wb[1] >> 7) : (wb[1] & 0x80)) * xt[5];
          accum += (wb[2] & 0x03) * (xt[5] * (Q3_SHIFTED ? 2.0f : 256.0f));
          accum += (Q3_SHIFTED ? ((wb[2] >> 2) & 0x07) : (wb[2] & 0x1c)) * xt[6];
          accum += (Q3_SHIFTED ? (wb[2] >> 5) : (wb[2] & 0xe0)) * xt[7];
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
          accum += (Q3_SHIFTED ? ((packed[0] >> 3) & 0x07) : (packed[0] & 0x38)) * xt[1];
          accum += (Q3_SHIFTED ? (packed[0] >> 6) : (packed[0] & 0xc0)) * xt[2];
          accum += (packed[1] & 0x01) * (xt[2] * (Q3_SHIFTED ? 4.0f : 256.0f));
          accum += (Q3_SHIFTED ? ((packed[1] >> 1) & 0x07) : (packed[1] & 0x0e)) * xt[3];
          accum += (Q3_SHIFTED ? ((packed[1] >> 4) & 0x07) : (packed[1] & 0x70)) * xt[4];
          accum += (Q3_SHIFTED ? (packed[1] >> 7) : (packed[1] & 0x80)) * xt[5];
          accum += (packed[2] & 0x03) * (xt[5] * (Q3_SHIFTED ? 2.0f : 256.0f));
          accum += (Q3_SHIFTED ? ((packed[2] >> 2) & 0x07) : (packed[2] & 0x1c)) * xt[6];
          accum += (Q3_SHIFTED ? (packed[2] >> 5) : (packed[2] & 0xe0)) * xt[7];
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


@lru_cache(maxsize=None)
def _target_verify_qmv_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "qwen3_5_target_verify_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QMV_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qargmax_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "qwen3_5_target_verify_qargmax_"
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
            "qwen3_5_target_verify_masked_qargmax_"
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
            "qwen3_5_target_verify_qmv_token_tiled_"
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
            "qwen3_5_target_verify_qargmax_token_tiled_"
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
            "qwen3_5_target_verify_masked_qargmax_token_tiled_"
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
            "qwen3_5_target_verify_qmv_streamed_"
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
            "qwen3_5_target_verify_qargmax_streamed_"
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
            "qwen3_5_target_verify_masked_qargmax_streamed_"
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
            "qwen3_5_target_verify_fused_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{shape_name}_{dtype_name}"
        ),
        input_names=input_names,
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size),
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
            "qwen3_5_target_verify_fused_qmv_streamed_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{shape_name}_{dtype_name}"
        ),
        input_names=input_names,
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size, 1),
        source=_target_verify_fused_qmv_source(
            _TARGET_VERIFY_QMV_STREAMED_SOURCE, n_sizes
        ),
    )


def _can_target_verify_quantized_head(linear) -> bool:
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


def _can_target_verify_quantized(linear, x: mx.array) -> bool:
    if (
        not _can_target_verify_quantized_head(linear)
        or x.ndim != 3
        or x.shape[1] < 1
        or x.dtype != linear.scales.dtype
    ):
        return False

    K = linear.weight.shape[1] * 32 // linear.bits
    return x.shape[-1] == K


def _target_verify_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    if not _can_target_verify_quantized(linear, x):
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]

    x = mx.contiguous(x)
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    results_per_simdgroup = 1 if streamed else 4
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


def _pad_token_mask_to_head(token_mask: mx.array, n_size: int) -> mx.array:
    """Widen a packed token bitmask to cover every lm_head output row.

    llguidance packs its bitmask over the tokenizer vocabulary, which can be
    smaller than the checkpoint's padded ``vocab_size`` (Qwen3.5 pads 248077
    real tokens up to 248320 rows). The masked argmax kernel indexes mask words
    by output row without bounds-checking, so the tail words have to exist, and
    have to read as disallowed so the padding rows can never be sampled.
    """
    required = (n_size + 31) // 32
    missing = required - token_mask.shape[1]
    if missing <= 0:
        return token_mask
    pad = mx.zeros((token_mask.shape[0], missing), dtype=token_mask.dtype)
    return mx.concatenate([token_mask, pad], axis=1)


def _target_verify_quantized_argmax(
    linear, x: mx.array, token_mask: Optional[mx.array] = None
) -> Optional[mx.array]:
    if not _can_target_verify_quantized(linear, x) or "bias" in linear:
        return None

    B, T, K = x.shape
    if T == 1 and 1 < B <= 4:
        out = _target_verify_quantized_argmax(
            linear, x.transpose(1, 0, 2), token_mask=token_mask
        )
        if out is not None:
            return out.transpose(1, 0)

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


def _target_verify_timewise(fn, x: mx.array) -> mx.array:
    return mx.concatenate([fn(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)


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
        return _target_verify_timewise(linear, x)

    if isinstance(linear, nn.Linear) and "bias" not in linear:
        out = _target_verify_weight(linear.weight, x)
        if out is not None:
            return out

    return _target_verify_singletons(linear, x)


def _target_verify_quantized_linears(linears, x: mx.array):
    if (
        not 2 <= len(linears) <= 4
        or x.ndim != 3
        or not 1 < x.shape[1] <= 8
        or not all(
            isinstance(linear, nn.QuantizedLinear)
            and linear.bits in (3, 4)
            and linear.bits == linears[0].bits
            and linear.group_size == linears[0].group_size
            and linear.mode == linears[0].mode
            and "bias" not in linear
            and _can_target_verify_quantized(linear, x)
            for linear in linears
        )
    ):
        return None

    B, T, K = x.shape
    n_sizes = tuple(int(linear.weight.shape[0]) for linear in linears)
    total_n = sum(n_sizes)
    x = mx.contiguous(x)
    bits = linears[0].bits
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
    out = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(total_n)),
        ],
        grid=(32, 2 * (total_n // (2 if streamed else 8)), B),
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


def _target_verify_linears(linears, x: mx.array):
    if not (
        x.ndim == 3
        and x.shape[1] > 1
        and all(
            isinstance(linear, (nn.Linear, nn.QuantizedLinear)) for linear in linears
        )
    ):
        from .language import _decode_quantized_linears_fused

        out = _decode_quantized_linears_fused(linears, x)
        if out is not None:
            return out
        return tuple(linear(x) for linear in linears)

    out = _target_verify_quantized_linears(linears, x)
    if out is not None:
        return out
    return tuple(_target_verify_linear(linear, x) for linear in linears)


def _target_verify_embedding_as_linear(embedding, x: mx.array):
    if not (x.ndim == 3 and x.shape[1] > 1):
        return embedding.as_linear(x)

    out = _target_verify_weight(embedding.weight, x)
    if out is not None:
        return out

    return _target_verify_timewise(embedding.as_linear, x)


class Qwen3_5ExactSpeculativeVerifier:
    """Run Qwen3.5 block verification with singleton-equivalent numerics."""

    @staticmethod
    def _helpers():
        # Imported lazily because language.py owns the shared cache and ragged
        # attention utilities and imports this verifier at module load time.
        from . import language

        return language

    def _linear(self, linear, x: mx.array) -> mx.array:
        return _target_verify_linear(linear, x)

    def _linears(self, linears, x: mx.array):
        return _target_verify_linears(linears, x)

    def _embedding_as_linear(self, embedding, x: mx.array) -> mx.array:
        return _target_verify_embedding_as_linear(embedding, x)

    def quantized_linear(self, linear, x: mx.array) -> Optional[mx.array]:
        return _target_verify_quantized_linear(linear, x)

    def quantized_argmax(
        self,
        linear,
        x: mx.array,
        token_mask: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        return _target_verify_quantized_argmax(linear, x, token_mask=token_mask)

    def can_quantized_head(self, linear) -> bool:
        return _can_target_verify_quantized_head(linear) and linear.bits != 3

    def pad_token_mask(self, token_mask: mx.array, n_size: int) -> mx.array:
        return _pad_token_mask_to_head(token_mask, n_size)

    def _attention(
        self,
        attention,
        x,
        mask,
        cache,
        position_ids,
        position_embeddings,
    ):
        batch, length, _ = x.shape
        q_proj_output, keys, values = self._linears(
            (attention.q_proj, attention.k_proj, attention.v_proj), x
        )
        queries, keys, values, gate, mask = attention._prepare_projected_qkv(
            q_proj_output,
            keys,
            values,
            cache,
            position_ids,
            position_embeddings,
            mask,
        )

        left_padded_decode = (
            mask == "left_padded_decode" if isinstance(mask, str) else False
        )
        if left_padded_decode:
            mask = None

        output = None
        if length > 1 or left_padded_decode:
            output = self._helpers()._qwen3_5_left_padded_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )

        if output is None and length > 1:
            prefix_length = kv_sequence_length(keys) - length
            output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        queries[:, :, index : index + 1, :],
                        slice_kv_sequence(keys, prefix_length + index + 1),
                        slice_kv_sequence(values, prefix_length + index + 1),
                        cache=cache,
                        scale=attention.scale,
                        mask=(
                            mask[
                                ...,
                                index : index + 1,
                                : prefix_length + index + 1,
                            ]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                    )
                    for index in range(length)
                ],
                axis=2,
            )
        elif output is None:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output * mx.sigmoid(gate))

    def _switch_glu(self, switch_mlp, x, indices):
        if x.ndim != 3 or x.shape[1] <= 1:
            return switch_mlp(x, indices)

        batch, length, width = x.shape
        top_k = indices.shape[-1]
        flat_x = x.reshape(batch * length, width)
        flat_indices = indices.reshape(batch * length, top_k)
        flat_x = mx.expand_dims(flat_x, (-2, -3))

        up = switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
        gate = switch_mlp.gate_proj(flat_x, flat_indices, sorted_indices=False)
        output = switch_mlp.down_proj(
            switch_mlp.activation(up, gate),
            flat_indices,
            sorted_indices=False,
        )
        return output.squeeze(-2).reshape(batch, length, top_k, -1)

    def _feed_forward(self, feed_forward, x):
        if hasattr(feed_forward, "switch_mlp"):
            gates = mx.softmax(
                self._linear(feed_forward.gate, x),
                axis=-1,
                precise=True,
            )
            top_k = feed_forward.top_k
            indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
            scores = mx.take_along_axis(gates, indices, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)

            output = self._switch_glu(feed_forward.switch_mlp, x, indices)
            output = (output * scores[..., None]).sum(axis=-2)

            shared_output = self._feed_forward(feed_forward.shared_expert, x)
            shared_output = (
                mx.sigmoid(self._linear(feed_forward.shared_expert_gate, x))
                * shared_output
            )
            return output + shared_output

        if all(
            hasattr(feed_forward, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            gate, up = self._linears((feed_forward.gate_proj, feed_forward.up_proj), x)
            return self._linear(feed_forward.down_proj, swiglu(gate, up))

        return feed_forward(x)

    @staticmethod
    def _normalize_gated_delta_qk(layer, q, k):
        del layer
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k

    def _gated_delta(self, layer, inputs, mask, cache, gdn_sink):
        helpers = self._helpers()
        batch, length, _ = inputs.shape
        mixed_qkv, z, b, a = self._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        z = z.reshape(batch, length, -1, layer.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, layer.conv_kernel_size - 1, layer.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != batch:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = layer.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_output = nn.silu(layer.conv1d(conv_input))
        q, k, v = [
            value.reshape(batch, length, heads, width)
            for value, heads, width in zip(
                mx.split(conv_output, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != batch:
            state = None
        q, k = self._normalize_gated_delta_qk(layer, q, k)

        initial_state = state
        output, state, intermediate_states = gated_delta_update_with_states(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            state,
            mask,
            use_kernel=not layer.training,
            state_steps=length - 1,
        )
        gdn_sink.append(
            (
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                initial_state,
                mask,
                conv_input,
                layer.conv_kernel_size,
                intermediate_states,
            )
        )

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(length)
                helpers._qwen3_5_advance_left_padding_info(cache, length)
                helpers._qwen3_5_advance_lengths_info(cache, length)

        output = layer.norm(output, z)
        return self._linear(layer.out_proj, output.reshape(batch, length, -1))

    def _layer(
        self,
        layer,
        hidden,
        mask,
        cache,
        position_ids,
        position_embeddings,
        gdn_sink,
    ):
        normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual = self._gated_delta(
                layer.linear_attn,
                normed,
                mask,
                cache,
                gdn_sink,
            )
        else:
            residual = self._attention(
                layer.self_attn,
                normed,
                mask,
                cache,
                position_ids,
                position_embeddings,
            )
        hidden = hidden + residual
        return hidden + self._feed_forward(
            layer.mlp,
            layer.post_attention_layernorm(hidden),
        )

    def _model(
        self,
        model,
        inputs,
        cache,
        inputs_embeds,
        position_ids,
        capture_layer_ids,
        hidden_sink,
        gdn_sink,
    ):
        helpers = self._helpers()
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(model.layers)

        fa_mask = helpers._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = helpers._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        decode_left_padding = (
            getattr(cache[model.fa_idx], "_qwen3_5_decode_left_padding", None)
            if isinstance(fa_mask, str) and fa_mask == "left_padded_decode"
            else None
        )
        helpers._set_qwen3_5_decode_left_padding(
            cache, model.layers, decode_left_padding
        )

        position_embeddings = None
        if position_ids is not None:
            for layer in model.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        position_embeddings = layer.self_attn.rotary_emb(
                            hidden, position_ids
                        )
                    break

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(
                layer,
                hidden,
                layer_mask,
                layer_cache,
                position_ids,
                position_embeddings,
                gdn_sink,
            )
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)

        return model.norm(hidden)

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache: Any = None,
        inputs_embeds: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        hidden_sink: list[mx.array] | None = (
            [] if capture_layer_ids is not None else None
        )
        gdn_sink: list = []
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            inputs_embeds,
            position_ids,
            capture_layer_ids,
            hidden_sink,
            gdn_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(hidden)

        if skip_logits:
            logits = None
        elif language_model.args.tie_word_embeddings:
            logits = self._embedding_as_linear(
                language_model.model.embed_tokens, hidden
            )
        else:
            logits = self._linear(language_model.lm_head, hidden)

        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={} if return_shared_kv else None,
        )


__all__ = ["Qwen3_5ExactSpeculativeVerifier"]
