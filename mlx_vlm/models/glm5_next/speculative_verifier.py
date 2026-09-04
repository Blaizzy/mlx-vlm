from functools import lru_cache
from typing import Any, Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention
from ..cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    CacheList,
    KVCache,
    PoolingCache,
)
from ..deepseek_v4.hyper_connection import _hc_kernel, hc_expand
from ..exact_speculative_verify import exact_speculative_verify_weight
from ..gated_delta import gated_delta_update
from ..qwen3_5.speculative_verifier import (
    _target_verify_linear,
    _target_verify_quantized_argmax,
)
from ..switch_layers import QuantizedSwitchLinear

_COMMON_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
"""


_HC_NORM_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;

constexpr int MIX = (2 + HC) * HC;
constexpr int BASE_OFF = 2 * HC;
constexpr float HC_EPS = HC_EPS_INT * 1e-9;
constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
constexpr int D4 = D / 4;

const device float* mix = mixes + row * MIX;
device float* post_out = post + row * HC;
device float* comb_out = comb + row * HC * HC;
threadgroup float pre_shared[HC];
threadgroup float local_inv_mean[1];
threadgroup float local_sums[32];

if (sg == 0) {
  const float pre_scale = scale[0];
  const float post_scale = scale[1];
  const float comb_scale = scale[2];
  const float active = lane < (uint)HC ? 1.0f : 0.0f;
  const uint llane = metal::min(lane, (uint)(HC - 1));

  float pre_z = mix[llane] * pre_scale + base[llane];
  float post_z = mix[HC + llane] * post_scale + base[HC + llane];
  float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
  float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
  if (lane < (uint)HC) {
    pre_shared[lane] = pre_v;
    post_out[lane] = post_v;
  }

  float4 value =
      (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale +
       *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
  float row_max = metal::max(
      metal::max(value.x, value.y), metal::max(value.z, value.w));
  float4 exponent = metal::fast::exp(value - row_max) * active;
  float4 result = exponent *
          (1.0f /
           (exponent.x + exponent.y + exponent.z + exponent.w + HC_EPS)) +
      HC_EPS * active;
  float4 column_inv = 1.0f /
      (float4(
           simd_sum(result.x), simd_sum(result.y),
           simd_sum(result.z), simd_sum(result.w)) +
       HC_EPS);
  result *= column_inv;
  for (int iter = 1; iter < ITERS; ++iter) {
    result *=
        (1.0f / (result.x + result.y + result.z + result.w + HC_EPS)) *
        active;
    column_inv = 1.0f /
        (float4(
             simd_sum(result.x), simd_sum(result.y),
             simd_sum(result.z), simd_sum(result.w)) +
         HC_EPS);
    result *= column_inv;
  }
  if (lane < (uint)HC) {
    *(device float4*)(comb_out + lane * HC) = result;
  }
}

if (sg == 0) {
  local_sums[lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const device T* x_row = (const device T*)x_in + row * HC * D;
using T4 = vec<T, 4>;
const device T4* x0 = (const device T4*)(x_row);
const device T4* x1 = (const device T4*)(x_row + D);
const device T4* x2 = (const device T4*)(x_row + 2 * D);
const device T4* x3 = (const device T4*)(x_row + 3 * D);

T4 rounded = T4(0);
float accum = 0.0f;
if (tid < D4) {
  float4 collapsed = fma(
      float4(pre_shared[0]), float4(x0[tid]),
      fma(
          float4(pre_shared[1]), float4(x1[tid]),
          fma(
              float4(pre_shared[2]), float4(x2[tid]),
              float4(pre_shared[3]) * float4(x3[tid]))));
  rounded = T4(collapsed);
  float4 rounded_float = float4(rounded);
  accum += rounded_float.x * rounded_float.x;
  accum += rounded_float.y * rounded_float.y;
  accum += rounded_float.z * rounded_float.z;
  accum += rounded_float.w * rounded_float.w;
}

accum = simd_sum(accum);
if (lane == 0) {
  local_sums[sg] = accum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  accum = simd_sum(local_sums[lane]);
  if (lane == 0) {
    local_inv_mean[0] = metal::precise::rsqrt(accum / D + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < D4) {
  float inv = local_inv_mean[0];
  const device T4* weights = (const device T4*)norm_weight;
  T4 scaled = T4(float4(rounded) * inv);
  T4 weight = weights[tid];
  ((device T4*)normalized)[row * D4 + tid] = T4(
      weight.x * scaled.x,
      weight.y * scaled.y,
      weight.z * scaled.z,
      weight.w * scaled.w);
}
"""


_HC_MIX_GEMV_SOURCE = r"""
uint out_block = threadgroup_position_in_grid.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

constexpr int TM = 4;
constexpr int TN = 4;
constexpr int SN = 32;
constexpr int BN = 8;
constexpr int BLOCK_N = BN * SN * TN;

int out_row = int(out_block) * TM;
int bn = (int(simd_gid) * SN + int(lane)) * TN;
float result[VERIFY_T][TM] = {0.0f};

for (int k = 0; k < K_SIZE; k += BLOCK_N) {
  float vectors[VERIFY_T][TN];
  for (int t = 0; t < VERIFY_T; ++t) {
    const device float* source = x + t * K_SIZE + k + bn;
    for (int tn = 0; tn < TN; ++tn) {
      vectors[t][tn] = source[tn];
    }
  }

  for (int tm = 0; tm < TM; ++tm) {
    const device W* source = weight + (out_row + tm) * K_SIZE + k + bn;
    W values[TN];
    for (int tn = 0; tn < TN; ++tn) {
      values[tn] = source[tn];
    }
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int tn = 0; tn < TN; ++tn) {
        result[t][tm] += values[tn] * vectors[t][tn];
      }
    }
  }
}

for (int t = 0; t < VERIFY_T; ++t) {
  for (int tm = 0; tm < TM; ++tm) {
    for (ushort offset = SN / 2; offset >= 1; offset >>= 1) {
      result[t][tm] += simd_shuffle_down(result[t][tm], offset);
    }
  }
}

threadgroup float partials[VERIFY_T][BN][TM];
if (lane == 0) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int tm = 0; tm < TM; ++tm) {
      partials[t][simd_gid][tm] = result[t][tm];
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0 && lane == 0) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int tm = 0; tm < TM; ++tm) {
      float value = partials[t][0][tm];
      for (int group = 1; group < BN; ++group) {
        value += partials[t][group][tm];
      }
      out[t * OUT_SIZE + out_row + tm] = value;
    }
  }
}
"""


_HC_NORMALIZED_NORM_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;

constexpr int MIX = (2 + HC) * HC;
constexpr int BASE_OFF = 2 * HC;
constexpr int K_SIZE = HC * D;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int MIX_SIMDS = 8;
constexpr int MIX_BLOCK = MIX_SIMDS * 32 * TN;
constexpr int MIX_BLOCKS = (MIX + TM - 1) / TM;
constexpr float HC_EPS = HC_EPS_INT * 1e-9;
constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
constexpr int D4 = D / 4;

const device T* x_row = x_in + row * K_SIZE;

// Match the standalone FP32 RMS reduction: 32 SIMD groups, four values per
// lane, then a fixed SIMD-group reduction over the 32 partials.
threadgroup float norm_partials[32];
threadgroup float inv_norm[1];
float norm_sum = 0.0f;
for (int k = int(tid) * TN; k < K_SIZE; k += 32 * 32 * TN) {
  float4 value = float4(*(const device vec<T, 4>*)(x_row + k));
  norm_sum += value.x * value.x;
  norm_sum += value.y * value.y;
  norm_sum += value.z * value.z;
  norm_sum += value.w * value.w;
}
norm_sum = simd_sum(norm_sum);
if (lane == 0) {
  norm_partials[sg] = norm_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  norm_sum = simd_sum(norm_partials[lane]);
  if (lane == 0) {
    inv_norm[0] = metal::precise::rsqrt(norm_sum / K_SIZE + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Six four-output GEMVs use the same eight-SIMD reduction tree as the
// standalone decode-exact mix kernel. Four blocks run in phase zero and the
// remaining two in phase one, sharing this threadgroup and the normalized x.
threadgroup float mix_partials[4][MIX_SIMDS][TM];
threadgroup float mix_values[MIX];
uint mix_slot = sg / MIX_SIMDS;
uint mix_sg = sg % MIX_SIMDS;
for (int phase = 0; phase < 2; ++phase) {
  int out_block = phase * 4 + int(mix_slot);
  if (out_block < MIX_BLOCKS) {
    int out_row = out_block * TM;
    int bn = (int(mix_sg) * 32 + int(lane)) * TN;
    float result[TM] = {0.0f};
    for (int k = 0; k < K_SIZE; k += MIX_BLOCK) {
      float vectors[TN];
      float inv = inv_norm[0];
      const device T* source_x = x_row + k + bn;
      for (int tn = 0; tn < TN; ++tn) {
        vectors[tn] = float(source_x[tn]) * inv;
      }
      for (int tm = 0; tm < TM; ++tm) {
        const device W* source_w =
            mix_weight + (out_row + tm) * K_SIZE + k + bn;
        for (int tn = 0; tn < TN; ++tn) {
          result[tm] += source_w[tn] * vectors[tn];
        }
      }
    }
    for (int tm = 0; tm < TM; ++tm) {
      for (ushort offset = 16; offset >= 1; offset >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], offset);
      }
      if (lane == 0) {
        mix_partials[mix_slot][mix_sg][tm] = result[tm];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (mix_sg == 0 && lane == 0 && out_block < MIX_BLOCKS) {
    int out_row = out_block * TM;
    for (int tm = 0; tm < TM; ++tm) {
      float value = mix_partials[mix_slot][0][tm];
      for (int group = 1; group < MIX_SIMDS; ++group) {
        value += mix_partials[mix_slot][group][tm];
      }
      if (out_row + tm < MIX) {
        mix_values[out_row + tm] = value;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

device float* post_out = post + row * HC;
device float* comb_out = comb + row * HC * HC;
threadgroup float pre_shared[HC];
threadgroup float collapsed_sums[32];
threadgroup float collapsed_inv[1];

if (sg == 0) {
  const float pre_scale = scale[0];
  const float post_scale = scale[1];
  const float comb_scale = scale[2];
  const float active = lane < (uint)HC ? 1.0f : 0.0f;
  const uint llane = metal::min(lane, (uint)(HC - 1));

  float pre_z = mix_values[llane] * pre_scale + base[llane];
  float post_z = mix_values[HC + llane] * post_scale + base[HC + llane];
  float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
  float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
  if (lane < (uint)HC) {
    pre_shared[lane] = pre_v;
    post_out[lane] = post_v;
  }

  float4 value =
      (*(threadgroup float4*)(mix_values + BASE_OFF + llane * HC) *
           comb_scale +
       *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
  float row_max = metal::max(
      metal::max(value.x, value.y), metal::max(value.z, value.w));
  float4 exponent = metal::fast::exp(value - row_max) * active;
  float4 result = exponent *
          (1.0f /
           (exponent.x + exponent.y + exponent.z + exponent.w + HC_EPS)) +
      HC_EPS * active;
  float4 column_inv = 1.0f /
      (float4(
           simd_sum(result.x), simd_sum(result.y),
           simd_sum(result.z), simd_sum(result.w)) +
       HC_EPS);
  result *= column_inv;
  for (int iter = 1; iter < ITERS; ++iter) {
    result *=
        (1.0f / (result.x + result.y + result.z + result.w + HC_EPS)) *
        active;
    column_inv = 1.0f /
        (float4(
             simd_sum(result.x), simd_sum(result.y),
             simd_sum(result.z), simd_sum(result.w)) +
         HC_EPS);
    result *= column_inv;
  }
  if (lane < (uint)HC) {
    *(device float4*)(comb_out + lane * HC) = result;
  }
  collapsed_sums[lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

using T4 = vec<T, 4>;
const device T4* x0 = (const device T4*)(x_row);
const device T4* x1 = (const device T4*)(x_row + D);
const device T4* x2 = (const device T4*)(x_row + 2 * D);
const device T4* x3 = (const device T4*)(x_row + 3 * D);
T4 rounded = T4(0);
float collapsed_sum = 0.0f;
if (tid < D4) {
  float4 collapsed = fma(
      float4(pre_shared[0]), float4(x0[tid]),
      fma(
          float4(pre_shared[1]), float4(x1[tid]),
          fma(
              float4(pre_shared[2]), float4(x2[tid]),
              float4(pre_shared[3]) * float4(x3[tid]))));
  rounded = T4(collapsed);
  float4 rounded_float = float4(rounded);
  collapsed_sum += rounded_float.x * rounded_float.x;
  collapsed_sum += rounded_float.y * rounded_float.y;
  collapsed_sum += rounded_float.z * rounded_float.z;
  collapsed_sum += rounded_float.w * rounded_float.w;
}
collapsed_sum = simd_sum(collapsed_sum);
if (lane == 0) {
  collapsed_sums[sg] = collapsed_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  collapsed_sum = simd_sum(collapsed_sums[lane]);
  if (lane == 0) {
    collapsed_inv[0] =
        metal::precise::rsqrt(collapsed_sum / D + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < D4) {
  float inv = collapsed_inv[0];
  const device T4* weights = (const device T4*)norm_weight;
  T4 scaled = T4(float4(rounded) * inv);
  T4 weight = weights[tid];
  ((device T4*)normalized)[row * D4 + tid] = T4(
      weight.x * scaled.x,
      weight.y * scaled.y,
      weight.z * scaled.z,
      weight.w * scaled.w);
}
"""


_DENSE_BLOCK_GEMV_SOURCE = r"""
uint tile = threadgroup_position_in_grid.y * SIMDS +
    simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int BLOCK_N = 32 * TN;
int out_row = int(tile) * TM;
int column = int(lane) * TN;
float result[BATCH_SIZE][VERIFY_T][TM] = {0.0f};

for (int k = 0; k < K_SIZE; k += BLOCK_N) {
  float vectors[BATCH_SIZE][VERIFY_T][TN];
  for (int b = 0; b < BATCH_SIZE; ++b) {
    for (int t = 0; t < VERIFY_T; ++t) {
      const device T* source =
          x + (b * VERIFY_T + t) * K_SIZE + k + column;
      for (int tn = 0; tn < TN; ++tn) {
        vectors[b][t][tn] = float(source[tn]);
      }
    }
  }
  for (int tm = 0; tm < TM; ++tm) {
    const device T* source = weight + (out_row + tm) * K_SIZE + k + column;
    T values[TN];
    for (int tn = 0; tn < TN; ++tn) {
      values[tn] = source[tn];
    }
    for (int b = 0; b < BATCH_SIZE; ++b) {
      for (int t = 0; t < VERIFY_T; ++t) {
        for (int tn = 0; tn < TN; ++tn) {
          result[b][t][tm] += values[tn] * vectors[b][t][tn];
        }
      }
    }
  }
}

for (int b = 0; b < BATCH_SIZE; ++b) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int tm = 0; tm < TM; ++tm) {
      for (ushort offset = 16; offset >= 1; offset >>= 1) {
        result[b][t][tm] += simd_shuffle_down(result[b][t][tm], offset);
      }
    }
  }
}
if (lane == 0) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int tm = 0; tm < TM; ++tm) {
        out[(b * VERIFY_T + t) * OUT_SIZE + out_row + tm] =
            T(result[b][t][tm]);
      }
    }
  }
}
"""


_HC_EXPAND_SOURCE = r"""
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
    volatile float product =
        post[row * HC + matrix_row] * float(x[row * D + column]);
    values[element] = product + values[element];
    out[row * HC * D + matrix_row * D + column] = T(values[element]);
  }
}
"""


_AFFINE5_HEADER = _COMMON_HEADER + r"""
inline void decode_affine5(
    const device uint8_t* w,
    float scale,
    float bias,
    thread float* out) {
  uint8_t w0 = w[0];
  uint8_t w1 = w[1];
  uint8_t w2 = w[2];
  uint8_t w3 = w[3];
  uint8_t w4 = w[4];
  out[0] = (w0 & 0x1f) * scale + bias;
  out[1] = (((w0 & 0xe0) >> 5) + ((w1 & 0x03) << 3)) * scale + bias;
  out[2] = ((w1 & 0x7c) >> 2) * scale + bias;
  out[3] = (((w1 & 0x80) >> 7) + ((w2 & 0x0f) << 1)) * scale + bias;
  out[4] = (((w2 & 0xf0) >> 4) + ((w3 & 0x01) << 4)) * scale + bias;
  out[5] = ((w3 & 0x3e) >> 1) * scale + bias;
  out[6] = (((w3 & 0xc0) >> 6) + ((w4 & 0x07) << 2)) * scale + bias;
  out[7] = ((w4 & 0xf8) >> 3) * scale + bias;
}
"""


_AFFINE4_HEADER = _COMMON_HEADER + r"""
inline void decode_affine4(
    const device uint8_t* w,
    float scale,
    float bias,
    thread float* out) {
  uint8_t w0 = w[0];
  uint8_t w1 = w[1];
  uint8_t w2 = w[2];
  uint8_t w3 = w[3];
  out[0] = (w0 & 0x0f) * scale + bias;
  out[1] = ((w0 & 0xf0) >> 4) * scale + bias;
  out[2] = (w1 & 0x0f) * scale + bias;
  out[3] = ((w1 & 0xf0) >> 4) * scale + bias;
  out[4] = (w2 & 0x0f) * scale + bias;
  out[5] = ((w2 & 0xf0) >> 4) * scale + bias;
  out[6] = (w3 & 0x0f) * scale + bias;
  out[7] = ((w3 & 0xf0) >> 4) * scale + bias;
}
"""


_AFFINE5_WIDE_SOURCE = r"""
uint tile = threadgroup_position_in_grid.y;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

constexpr int K_LANES = 8;
constexpr int ROWS_PER_SIMD = 32 / K_LANES;
constexpr int ROWS_PER_TG = 2 * ROWS_PER_SIMD;
constexpr int SUB = 8;

short k_lane = simd_lid % K_LANES;
short sg_row = simd_lid / K_LANES;
int out_row = int(tile) * ROWS_PER_TG + int(simd_gid) * ROWS_PER_SIMD + sg_row;
int row = min(out_row, N_SIZE - 1);

constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
const device uint8_t* wrow = (const device uint8_t*)w + row * W_ROW_BYTES;
const device T* srow = scales + row * GROUPS;
const device T* brow = biases + row * GROUPS;

float result[VERIFY_T][BATCH_SIZE];
for (int t = 0; t < VERIFY_T; ++t) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    result[t][b] = 0.0f;
  }
}

for (int g = k_lane; g < GROUPS; g += K_LANES) {
  float scale = float(srow[g]);
  float bias = float(brow[g]);
  for (int sc = 0; sc < GROUP_SIZE / SUB; ++sc) {
    int k0 = g * GROUP_SIZE + sc * SUB;
    const device uint8_t* wc = wrow + k0 * 5 / 8;
    float wdq[SUB];
    decode_affine5(wc, scale, bias, wdq);
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int b = 0; b < BATCH_SIZE; ++b) {
        const device T* xc = x + (b * VERIFY_T + t) * K_SIZE + k0;
        float accum = 0.0f;
        for (int i = 0; i < SUB; ++i) {
          accum += float(xc[i]) * wdq[i];
        }
        result[t][b] += accum;
      }
    }
  }
}

for (int t = 0; t < VERIFY_T; ++t) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    result[t][b] += simd_shuffle_down(result[t][b], 4);
    result[t][b] += simd_shuffle_down(result[t][b], 2);
    result[t][b] += simd_shuffle_down(result[t][b], 1);
  }
}

if (k_lane == 0 && out_row < N_SIZE) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      y[(b * VERIFY_T + t) * N_SIZE + out_row] = T(result[t][b]);
    }
  }
}
"""


_AFFINE5_WIDE_ARGMAX_SOURCE = _AFFINE5_WIDE_SOURCE.replace(
    r"""if (k_lane == 0 && out_row < N_SIZE) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      y[(b * VERIFY_T + t) * N_SIZE + out_row] = T(result[t][b]);
    }
  }
}
""",
    r"""threadgroup float simd_best_values[VERIFY_T][BATCH_SIZE][2];
threadgroup int simd_best_indices[VERIFY_T][BATCH_SIZE][2];

for (int t = 0; t < VERIFY_T; ++t) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    float rounded = float(T(result[t][b]));
    float best = simd_shuffle(rounded, 0);
    int best_index = int(tile) * ROWS_PER_TG + int(simd_gid) * ROWS_PER_SIMD;
    for (int row = 1; row < ROWS_PER_SIMD; ++row) {
      float candidate = simd_shuffle(rounded, row * K_LANES);
      int candidate_index =
          int(tile) * ROWS_PER_TG + int(simd_gid) * ROWS_PER_SIMD + row;
      if (candidate > best) {
        best = candidate;
        best_index = candidate_index;
      }
    }
    if (simd_lid == 0) {
      simd_best_values[t][b][simd_gid] = best;
      simd_best_indices[t][b][simd_gid] = best_index;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0 && simd_lid == 0) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      float best = simd_best_values[t][b][0];
      int best_index = simd_best_indices[t][b][0];
      if (simd_best_values[t][b][1] > best) {
        best = simd_best_values[t][b][1];
        best_index = simd_best_indices[t][b][1];
      }
      int offset = (b * VERIFY_T + t) * NUM_TILES + int(tile);
      tile_values[offset] = T(best);
      tile_indices[offset] = best_index;
    }
  }
}
""",
)


_AFFINE5_SWITCH_HEADER = _COMMON_HEADER + r"""
constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int BYTES_PER_PACK = 5;
constant constexpr int PACKS_PER_THREAD = __PACKS_PER_THREAD__;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD =
    __GROUP_SIZE__ / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = __RESULTS_PER_SIMDGROUP__;
constant constexpr int NUM_SIMDGROUPS = __NUM_SIMDGROUPS__;
constant constexpr int ROWS_PER_TG =
    RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

template <typename T>
inline float load_affine5_vector_exact(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
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
  return sum;
}

inline float affine5_qdot_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD / 8; ++i) {
    const thread float* xt = x_thread + 8 * i;
    const device uint8_t* wb = w + 5 * i;
    accum += (wb[0] & 0x1f) * xt[0];
    accum += (wb[0] & 0xe0) * xt[1];
    accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
    accum += (wb[1] & 0x7c) * xt[2];
    accum += (wb[1] & 0x80) * xt[3];
    accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
    accum += (wb[2] & 0xf0) * xt[4];
    accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
    accum += (wb[3] & 0x3e) * xt[5];
    accum += (wb[3] & 0xc0) * xt[6];
    accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
    accum += (wb[4] & 0xf8) * xt[7];
  }
  return scale * accum + sum * bias;
}
"""


_AFFINE4_SWITCH_HEADER = _COMMON_HEADER + r"""
constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int PACKS_PER_THREAD = __PACKS_PER_THREAD__;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD =
    __GROUP_SIZE__ / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = __RESULTS_PER_SIMDGROUP__;
constant constexpr int NUM_SIMDGROUPS = __NUM_SIMDGROUPS__;
constant constexpr int ROWS_PER_TG =
    RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

template <typename T>
inline float load_affine4_vector_exact(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float affine4_qdot_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
  float sum) {
  float accum = 0.0f;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""


def _affine_source(source: str, bits: int) -> str:
    if bits == 5:
        return source
    if bits != 4:
        raise ValueError(f"Unsupported affine bit width: {bits}")
    return source.replace("affine5", "affine4").replace("* 5 / 8", "* 4 / 8")


_AFFINE5_SWITCH_GATE_UP_SOURCE = r"""
uint n_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
int token = int(route) / TOP_K;
int route_index = int(route) - token * TOP_K;
int expert = int(indices[route]);
int paired_route = -1;
if (token == 0) {
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[TOP_K + other]) == expert) {
      paired_route = TOP_K + other;
      break;
    }
  }
} else {
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[other]) == expert) {
      return;
    }
  }
}
int pair_count = paired_route >= 0 ? 2 : 1;
constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;

const device uint8_t* up_ws = (const device uint8_t*)up_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* up_sc = up_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* up_bs = up_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device uint8_t* gate_ws = (const device uint8_t*)gate_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* gate_sc = gate_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* gate_bs = gate_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* xk[2];
xk[0] = x + token * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;
xk[1] = x + K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

float up_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float gate_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float x_thread[2][VALUES_PER_THREAD];

for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sums[2];
  for (int pair = 0; pair < pair_count; ++pair) {
    sums[pair] = load_affine5_vector_exact<T>(xk[pair], x_thread[pair]);
  }
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* uw = up_ws + row * W_ROW_BYTES;
    const device T* us = up_sc + row * GROUPS;
    const device T* ub = up_bs + row * GROUPS;
    const device uint8_t* gw = gate_ws + row * W_ROW_BYTES;
    const device T* gs = gate_sc + row * GROUPS;
    const device T* gb = gate_bs + row * GROUPS;
    for (int pair = 0; pair < pair_count; ++pair) {
      up_result[pair][row] += affine5_qdot_exact(
          uw, x_thread[pair], float(us[0]), float(ub[0]), sums[pair]);
      gate_result[pair][row] += affine5_qdot_exact(
          gw, x_thread[pair], float(gs[0]), float(gb[0]), sums[pair]);
    }
  }
  up_ws += BLOCK_SIZE * 5 / 8;
  up_sc += BLOCK_SIZE / GROUP_SIZE;
  up_bs += BLOCK_SIZE / GROUP_SIZE;
  gate_ws += BLOCK_SIZE * 5 / 8;
  gate_sc += BLOCK_SIZE / GROUP_SIZE;
  gate_bs += BLOCK_SIZE / GROUP_SIZE;
  xk[0] += BLOCK_SIZE;
  xk[1] += BLOCK_SIZE;
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  for (int pair = 0; pair < pair_count; ++pair) {
    float up_value = simd_sum(up_result[pair][row]);
    float gate_value = simd_sum(gate_result[pair][row]);
    if (simd_lid == 0 && n < N_SIZE) {
      int output_route = pair == 0 ? int(route) : paired_route;
      up_y[output_route * N_SIZE + n] = T(up_value);
      gate_y[output_route * N_SIZE + n] = T(gate_value);
    }
  }
}
"""


_AFFINE5_DENSE_BLOCK_SOURCE = r"""
uint n_tile = threadgroup_position_in_grid.y;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;

const device uint8_t* ws = (const device uint8_t*)w +
    out_row * W_ROW_BYTES + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* sc = scales + out_row * GROUPS +
    int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* bs = biases + out_row * GROUPS +
    int(simd_lid) / SCALE_STEP_PER_THREAD;

float result[VERIFY_T][RESULTS_PER_SIMDGROUP] = {0.0f};
float x_thread[VERIFY_T][VALUES_PER_THREAD];
const device T* xk = x + int(simd_lid) * VALUES_PER_THREAD;

for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sums[VERIFY_T];
  bool active = k + int(simd_lid) * VALUES_PER_THREAD < K_SIZE;
  for (int token = 0; token < VERIFY_T; ++token) {
    if (active) {
      sums[token] = load_affine5_vector_exact<T>(
          xk + token * K_SIZE,
          x_thread[token]);
    } else {
      sums[token] = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        x_thread[token][i] = 0.0f;
      }
    }
  }
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* wr = ws + row * W_ROW_BYTES;
    const device T* sr = sc + row * GROUPS;
    const device T* br = bs + row * GROUPS;
    for (int token = 0; token < VERIFY_T; ++token) {
      if (active) {
        result[token][row] += affine5_qdot_exact(
            wr,
            x_thread[token],
            float(sr[0]),
            float(br[0]),
            sums[token]);
      }
    }
  }
  ws += BLOCK_SIZE * 5 / 8;
  sc += BLOCK_SIZE / GROUP_SIZE;
  bs += BLOCK_SIZE / GROUP_SIZE;
  xk += BLOCK_SIZE;
}

for (int token = 0; token < VERIFY_T; ++token) {
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    int n = out_row + row;
    float value = simd_sum(result[token][row]);
    if (simd_lid == 0 && n < N_SIZE) {
      y[token * N_SIZE + n] = T(value);
    }
  }
}
"""


_AFFINE5_MOE_DOWN_SOURCE = r"""
uint n_tile = threadgroup_position_in_grid.y;
uint token = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;

T routed_sum[RESULTS_PER_SIMDGROUP] = {T(0)};
for (int route = 0; route < TOP_K; ++route) {
  int route_offset = int(token) * TOP_K + route;
  int expert = int(indices[route_offset]);
  const device uint8_t* ws = (const device uint8_t*)w +
      expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* sc = scales + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* bs = biases + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* xk = x + route_offset * K_SIZE +
      int(simd_lid) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];
  for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
    float sum = load_affine5_vector_exact<T>(xk, x_thread);
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      const device uint8_t* wr = ws + row * W_ROW_BYTES;
      const device T* sr = sc + row * GROUPS;
      const device T* br = bs + row * GROUPS;
      result[row] += affine5_qdot_exact(
          wr, x_thread, float(sr[0]), float(br[0]), sum);
    }
    ws += BLOCK_SIZE * 5 / 8;
    sc += BLOCK_SIZE / GROUP_SIZE;
    bs += BLOCK_SIZE / GROUP_SIZE;
    xk += BLOCK_SIZE;
  }

  T route_weight = T(route_weights[route_offset]);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    float value = simd_sum(result[row]);
    if (simd_lid == 0) {
      T product = T(T(value) * route_weight);
      routed_sum[row] = T(routed_sum[row] + product);
    }
  }
}

const device uint8_t* shared_ws = (const device uint8_t*)shared_w +
    out_row * W_ROW_BYTES + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* shared_sc = shared_scales + out_row * GROUPS +
    int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* shared_bs = shared_biases + out_row * GROUPS +
    int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* shared_xk = shared_x + int(token) * K_SIZE +
    int(simd_lid) * VALUES_PER_THREAD;
float shared_result[RESULTS_PER_SIMDGROUP] = {0.0f};
float shared_x_thread[VALUES_PER_THREAD];
for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sum = load_affine5_vector_exact<T>(shared_xk, shared_x_thread);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* wr = shared_ws + row * W_ROW_BYTES;
    const device T* sr = shared_sc + row * GROUPS;
    const device T* br = shared_bs + row * GROUPS;
    shared_result[row] += affine5_qdot_exact(
        wr, shared_x_thread, float(sr[0]), float(br[0]), sum);
  }
  shared_ws += BLOCK_SIZE * 5 / 8;
  shared_sc += BLOCK_SIZE / GROUP_SIZE;
  shared_bs += BLOCK_SIZE / GROUP_SIZE;
  shared_xk += BLOCK_SIZE;
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  float shared_value = simd_sum(shared_result[row]);
  if (simd_lid == 0 && n < N_SIZE) {
    y[int(token) * N_SIZE + n] = T(routed_sum[row] + T(shared_value));
  }
}
"""


_MXFP8_HEADER = _COMMON_HEADER + r"""
struct fp8_e4m3 {
  operator half() {
    uint16_t value = (bits & 127) << 7;
    half converted = as_type<half>(value);
    converted *= 256.0h;
    return (bits & 128) ? -converted : converted;
  }
  operator float() { return float(this->operator half()); }
  uint8_t bits;
};

inline float decode_e4m3(uint8_t value) {
  return float(*(thread fp8_e4m3*)(&value));
}

inline float decode_e8m0(uint8_t value) {
  uint32_t output = value == 0 ? 0x400000 : (uint32_t(value) << 23);
  return as_type<float>(output);
}
"""


_MXFP8_WIDE_SOURCE = r"""
uint tile = threadgroup_position_in_grid.y;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

constexpr int K_LANES = 16;
constexpr int ROWS_PER_SIMD = 32 / K_LANES;
constexpr int ROWS_PER_TG = 2 * ROWS_PER_SIMD;
constexpr int NF4 = GROUP_SIZE / 4;

short k_lane = simd_lid % K_LANES;
short sg_row = simd_lid / K_LANES;
int out_row = int(tile) * ROWS_PER_TG + int(simd_gid) * ROWS_PER_SIMD + sg_row;
int row = min(out_row, N_SIZE - 1);

constexpr int GROUPS = K_SIZE / GROUP_SIZE;
const device uint8_t* wrow = (const device uint8_t*)w + row * K_SIZE;
const device uint8_t* srow = scales + row * GROUPS;

float result[VERIFY_T][BATCH_SIZE];
for (int t = 0; t < VERIFY_T; ++t) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    result[t][b] = 0.0f;
  }
}

for (int g = k_lane; g < GROUPS; g += K_LANES) {
  int k0 = g * GROUP_SIZE;
  float scale = decode_e8m0(srow[g]);
  const device uint8_t* wg = wrow + k0;
  float4 w4[NF4];
  for (int i = 0; i < NF4; ++i) {
    w4[i] = float4(
        decode_e4m3(wg[4 * i]),
        decode_e4m3(wg[4 * i + 1]),
        decode_e4m3(wg[4 * i + 2]),
        decode_e4m3(wg[4 * i + 3]));
  }
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      const device vec<T, 4>* xv4 =
          (const device vec<T, 4>*)(x + (b * VERIFY_T + t) * K_SIZE + k0);
      float accum = 0.0f;
      for (int i = 0; i < NF4; ++i) {
        accum += dot(w4[i], float4(xv4[i]));
      }
      result[t][b] += scale * accum;
    }
  }
}

for (int t = 0; t < VERIFY_T; ++t) {
  for (int b = 0; b < BATCH_SIZE; ++b) {
    result[t][b] += simd_shuffle_down(result[t][b], 8);
    result[t][b] += simd_shuffle_down(result[t][b], 4);
    result[t][b] += simd_shuffle_down(result[t][b], 2);
    result[t][b] += simd_shuffle_down(result[t][b], 1);
  }
}

if (k_lane == 0 && out_row < N_SIZE) {
  for (int t = 0; t < VERIFY_T; ++t) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      y[(b * VERIFY_T + t) * N_SIZE + out_row] = T(result[t][b]);
    }
  }
}
"""


_MXFP8_SINGLE_SOURCE = r"""
uint tile = threadgroup_position_in_grid.y;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

constexpr int RESULTS = 4;
constexpr int ROWS_PER_TG = 2 * RESULTS;
constexpr int VALUES_PER_THREAD = 8;
constexpr int BLOCK_SIZE = 32 * VALUES_PER_THREAD;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;

int out_row = int(tile) * ROWS_PER_TG + int(simd_gid) * RESULTS;
float result[VERIFY_T][RESULTS];
for (int t = 0; t < VERIFY_T; ++t) {
  for (int row = 0; row < RESULTS; ++row) {
    result[t][row] = 0.0f;
  }
}

const device uint8_t* ws =
    (const device uint8_t*)w + out_row * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;
const device uint8_t* ss =
    scales + out_row * GROUPS + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);

for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float xt[VERIFY_T][VALUES_PER_THREAD];
  for (int t = 0; t < VERIFY_T; ++t) {
    const device T* source = x + t * K_SIZE + k + int(simd_lid) * VALUES_PER_THREAD;
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      xt[t][i] = float(source[i]);
    }
  }
  for (int row = 0; row < RESULTS; ++row) {
    const device uint8_t* wr = ws + row * K_SIZE;
    const device uint8_t* sr = ss + row * GROUPS;
    float scale = decode_e8m0(sr[0]);
    for (int t = 0; t < VERIFY_T; ++t) {
      float accum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        accum += xt[t][i] * decode_e4m3(wr[i]);
      }
      result[t][row] += scale * accum;
    }
  }
  ws += BLOCK_SIZE;
  ss += BLOCK_SIZE / GROUP_SIZE;
}

for (int t = 0; t < VERIFY_T; ++t) {
  for (int row = 0; row < RESULTS; ++row) {
    float value = simd_sum(result[t][row]);
    if (simd_lid == 0 && out_row + row < N_SIZE) {
      y[t * N_SIZE + out_row + row] = T(value);
    }
  }
}
"""


def _dtype_name(dtype) -> str:
    return {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unknown")


def _affine_exact_header(
    bits,
    group_size,
    results_per_simdgroup,
    num_simdgroups,
    packs_per_thread=2,
):
    header = _AFFINE5_SWITCH_HEADER if bits == 5 else _AFFINE4_SWITCH_HEADER
    return (
        header.replace("__GROUP_SIZE__", str(group_size))
        .replace("__RESULTS_PER_SIMDGROUP__", str(results_per_simdgroup))
        .replace("__NUM_SIMDGROUPS__", str(num_simdgroups))
        .replace("__PACKS_PER_THREAD__", str(packs_per_thread))
    )


@lru_cache(maxsize=None)
def _hc_norm_kernel(dtype, hc_mult, width, sinkhorn_iters, hc_eps, norm_eps):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_hc_norm_"
            f"{_dtype_name(dtype)}_h{hc_mult}_d{width}_i{sinkhorn_iters}_"
            f"he{round(hc_eps / 1e-9)}_ne{round(norm_eps / 1e-9)}"
        ),
        input_names=[
            "x_in",
            "mixes",
            "scale",
            "base",
            "norm_weight",
        ],
        output_names=["normalized", "post", "comb"],
        source=_HC_NORM_SOURCE,
    )


@lru_cache(maxsize=None)
def _hc_normalized_norm_kernel(
    dtype,
    weight_dtype,
    hc_mult,
    width,
    sinkhorn_iters,
    hc_eps,
    norm_eps,
):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_hc_normalized_norm_"
            f"{_dtype_name(dtype)}_{_dtype_name(weight_dtype)}_h{hc_mult}_"
            f"d{width}_i{sinkhorn_iters}_he{round(hc_eps / 1e-9)}_"
            f"ne{round(norm_eps / 1e-9)}"
        ),
        input_names=[
            "x_in",
            "mix_weight",
            "scale",
            "base",
            "norm_weight",
        ],
        output_names=["normalized", "post", "comb"],
        source=_HC_NORMALIZED_NORM_SOURCE,
    )


@lru_cache(maxsize=None)
def _fp32_decode_block_gemv_kernel(weight_dtype, length, k_size, out_size):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_fp32_decode_block_gemv_"
            f"{_dtype_name(weight_dtype)}_t{length}_k{k_size}_o{out_size}"
        ),
        input_names=["x", "weight"],
        output_names=["out"],
        header=_COMMON_HEADER,
        source=_HC_MIX_GEMV_SOURCE.replace(
            "constexpr int BN = 8;", "constexpr int BN = 1;"
        ),
    )


@lru_cache(maxsize=None)
def _dense_block_gemv_kernel(dtype, batch, length, k_size, out_size):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_dense_block_gemv_"
            f"{_dtype_name(dtype)}_b{batch}_t{length}_k{k_size}_o{out_size}"
        ),
        input_names=["x", "weight"],
        output_names=["out"],
        header=_COMMON_HEADER,
        source=_DENSE_BLOCK_GEMV_SOURCE,
    )


@lru_cache(maxsize=None)
def _hc_expand_kernel(dtype, rows, hc_mult, width):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_hc_expand_"
            f"{_dtype_name(dtype)}_r{rows}_h{hc_mult}_d{width}"
        ),
        input_names=["x", "residual", "post", "comb"],
        output_names=["out"],
        header=_COMMON_HEADER,
        source=_HC_EXPAND_SOURCE,
    )


@lru_cache(maxsize=None)
def _affine_wide_kernel(bits, dtype, batch, length, k_size, n_size, group_size):
    return mx.fast.metal_kernel(
        name=(
            f"glm5_next_verify_affine{bits}_wide_"
            f"{_dtype_name(dtype)}_b{batch}_t{length}_k{k_size}_n{n_size}_g{group_size}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_AFFINE5_HEADER if bits == 5 else _AFFINE4_HEADER,
        source=_affine_source(_AFFINE5_WIDE_SOURCE, bits),
    )


@lru_cache(maxsize=None)
def _affine_wide_argmax_kernel(
    bits,
    dtype,
    batch,
    length,
    k_size,
    n_size,
    group_size,
):
    return mx.fast.metal_kernel(
        name=(
            f"glm5_next_verify_affine{bits}_wide_argmax_"
            f"{_dtype_name(dtype)}_b{batch}_t{length}_k{k_size}_n{n_size}_g{group_size}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_AFFINE5_HEADER if bits == 5 else _AFFINE4_HEADER,
        source=_affine_source(_AFFINE5_WIDE_ARGMAX_SOURCE, bits),
    )


@lru_cache(maxsize=None)
def _affine_switch_gate_up_kernel(
    bits,
    dtype,
    length,
    k_size,
    n_size,
    top_k,
    group_size,
):
    return mx.fast.metal_kernel(
        name=(
            f"glm5_next_verify_affine{bits}_switch_gate_up_"
            f"{_dtype_name(dtype)}_t{length}_k{k_size}_n{n_size}_"
            f"e{top_k}_g{group_size}"
        ),
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
        output_names=["up_y", "gate_y"],
        header=_affine_exact_header(bits, group_size, 1, 4),
        source=_affine_source(_AFFINE5_SWITCH_GATE_UP_SOURCE, bits),
    )


@lru_cache(maxsize=None)
def _affine_dense_block_kernel(
    bits,
    dtype,
    length,
    k_size,
    n_size,
    group_size,
    packs_per_thread,
):
    return mx.fast.metal_kernel(
        name=(
            f"glm5_next_verify_affine{bits}_dense_block_"
            f"{_dtype_name(dtype)}_t{length}_k{k_size}_n{n_size}_"
            f"g{group_size}_p{packs_per_thread}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_affine_exact_header(bits, group_size, 1, 1, packs_per_thread),
        source=_affine_source(_AFFINE5_DENSE_BLOCK_SOURCE, bits),
    )


@lru_cache(maxsize=None)
def _affine_moe_down_kernel(
    bits,
    dtype,
    length,
    k_size,
    n_size,
    top_k,
    group_size,
):
    return mx.fast.metal_kernel(
        name=(
            f"glm5_next_verify_affine{bits}_moe_down_"
            f"{_dtype_name(dtype)}_t{length}_k{k_size}_n{n_size}_"
            f"e{top_k}_g{group_size}"
        ),
        input_names=[
            "x",
            "indices",
            "route_weights",
            "w",
            "scales",
            "biases",
            "shared_x",
            "shared_w",
            "shared_scales",
            "shared_biases",
        ],
        output_names=["y"],
        header=_affine_exact_header(bits, group_size, 2, 4),
        source=_affine_source(_AFFINE5_MOE_DOWN_SOURCE, bits),
    )


@lru_cache(maxsize=None)
def _mxfp8_wide_kernel(dtype, batch, length, k_size, n_size, group_size):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_mxfp8_wide_"
            f"{_dtype_name(dtype)}_b{batch}_t{length}_k{k_size}_n{n_size}_g{group_size}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["y"],
        header=_MXFP8_HEADER,
        source=_MXFP8_WIDE_SOURCE,
    )


@lru_cache(maxsize=None)
def _mxfp8_single_kernel(dtype, length, k_size, n_size, group_size):
    return mx.fast.metal_kernel(
        name=(
            "glm5_next_verify_mxfp8_single_"
            f"{_dtype_name(dtype)}_t{length}_k{k_size}_n{n_size}_g{group_size}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["y"],
        header=_MXFP8_HEADER,
        source=_MXFP8_SINGLE_SOURCE,
    )


def exact_dense_block_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Decode-exact B=1 dense projection with weight reuse across time."""
    if (
        not mx.metal.is_available()
        or not isinstance(linear, nn.Linear)
        or "bias" in linear
        or x.ndim != 3
        or not 1 <= x.shape[0] <= 4
        or not 1 < x.shape[1] <= 4
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.weight.dtype != x.dtype
        or linear.weight.size < 50_000_000
        or linear.weight.shape[1] != x.shape[-1]
        or x.shape[-1] % 128
        or linear.weight.shape[0] % 4
    ):
        return None

    batch, length, k_size = x.shape
    out_size = linear.weight.shape[0]
    x = mx.contiguous(x)
    kernel = _dense_block_gemv_kernel(x.dtype, batch, length, k_size, out_size)
    return kernel(
        inputs=[x, linear.weight],
        template=[
            ("T", x.dtype),
            ("BATCH_SIZE", int(batch)),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("OUT_SIZE", int(out_size)),
            ("SIMDS", 8),
        ],
        grid=(32, out_size // 4, 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, length, out_size)],
        output_dtypes=[x.dtype],
    )[0]


def exact_quantized_block_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Decode-exact BxT projection with weight reuse across verifier time."""
    if (
        not mx.metal.is_available()
        or not isinstance(linear, nn.QuantizedLinear)
        or x.ndim != 3
        or not 1 < x.shape[1] <= 4
        or not 1 <= x.shape[0] <= 4
        or x.dtype not in (mx.bfloat16, mx.float16)
    ):
        return None

    batch, length, k_size = x.shape
    n_size = linear.weight.shape[0]
    if k_size % linear.group_size or n_size < 8:
        return None

    x = mx.contiguous(x)
    template = [
        ("T", x.dtype),
        ("BATCH_SIZE", int(batch)),
        ("VERIFY_T", int(length)),
        ("K_SIZE", int(k_size)),
        ("N_SIZE", int(n_size)),
        ("GROUP_SIZE", int(linear.group_size)),
    ]

    if (
        linear.mode == "affine"
        and linear.bits in (4, 5)
        and linear.biases is not None
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and linear.group_size % 8 == 0
    ):
        if batch == 1 and not (k_size % 512 or n_size % 16):
            packs_per_thread = 2
            kernel = _affine_dense_block_kernel(
                linear.bits,
                x.dtype,
                length,
                k_size,
                n_size,
                linear.group_size,
                packs_per_thread,
            )
            output = kernel(
                inputs=[x, linear.weight, linear.scales, linear.biases],
                template=[
                    ("T", x.dtype),
                    ("VERIFY_T", int(length)),
                    ("K_SIZE", int(k_size)),
                    ("N_SIZE", int(n_size)),
                    ("GROUP_SIZE", int(linear.group_size)),
                ],
                grid=(32, 16 * (n_size // 16), 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(batch, length, n_size)],
                output_dtypes=[x.dtype],
            )[0]
            if "bias" in linear:
                output = output + linear["bias"]
            return output
        if batch == 1:
            return None
        # MLX uses a different short-vector reduction for decode-shaped
        # Bx1 affine projections. The wide verifier kernel follows the regular
        # group reduction and is therefore not bit exact below 512 inputs.
        # Let the verifier's batched MLX fallback preserve the B-row kernel.
        if k_size < 512:
            return None
        kernel = _affine_wide_kernel(
            linear.bits,
            x.dtype,
            batch,
            length,
            k_size,
            n_size,
            linear.group_size,
        )
        inputs = [x, linear.weight, linear.scales, linear.biases]
        rows_per_tg = 8
    elif (
        linear.mode == "mxfp8"
        and linear.bits == 8
        and linear.group_size == 32
        and linear.biases is None
        and linear.scales.dtype == mx.uint8
    ):
        if batch == 1:
            if k_size % 512 or n_size % 8:
                return None
            kernel = _mxfp8_single_kernel(
                x.dtype, length, k_size, n_size, linear.group_size
            )
            rows_per_tg = 8
        else:
            kernel = _mxfp8_wide_kernel(
                x.dtype, batch, length, k_size, n_size, linear.group_size
            )
            rows_per_tg = 4
        inputs = [x, linear.weight, linear.scales]
    else:
        return None

    output = kernel(
        inputs=inputs,
        template=template,
        grid=(32, 2 * ((n_size + rows_per_tg - 1) // rows_per_tg), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(batch, length, n_size)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        output = output + linear["bias"]
    return output


def exact_affine_switch_gate_up(switch, x: mx.array, indices: mx.array):
    """Project selected affine up/gate weights in one verifier dispatch."""
    up = getattr(switch, "up_proj", None)
    gate = getattr(switch, "gate_proj", None)
    linears = (up, gate)
    if (
        not mx.metal.is_available()
        or x.ndim != 3
        or x.shape[0] != 1
        or not 1 < x.shape[1] <= 4
        or indices.ndim != 3
        or indices.shape[:2] != x.shape[:2]
        or x.dtype not in (mx.bfloat16, mx.float16)
        or not all(
            isinstance(linear, QuantizedSwitchLinear)
            and linear.mode == "affine"
            and linear.bits in (4, 5)
            and linear.biases is not None
            and "bias" not in linear
            and linear.group_size == up.group_size
            and linear.bits == up.bits
            and linear.scales.dtype == x.dtype
            and linear.biases.dtype == x.dtype
            for linear in linears
        )
    ):
        return None

    _, length, k_size = x.shape
    top_k = indices.shape[-1]
    n_size = up.weight.shape[1]
    if (
        gate.weight.shape != up.weight.shape
        or k_size != up.input_dims
        or k_size % 512
        or n_size % 8
        or up.group_size % 16
    ):
        return None

    x = mx.contiguous(x)
    indices = mx.contiguous(indices.astype(mx.int32))
    kernel = _affine_switch_gate_up_kernel(
        up.bits,
        x.dtype,
        length,
        k_size,
        n_size,
        top_k,
        up.group_size,
    )
    return kernel(
        inputs=[
            x,
            indices,
            up.weight,
            up.scales,
            up.biases,
            gate.weight,
            gate.scales,
            gate.biases,
        ],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("N_SIZE", int(n_size)),
            ("TOP_K", int(top_k)),
            ("GROUP_SIZE", int(up.group_size)),
        ],
        grid=(32, 16 * (n_size // 16), length * top_k),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (1, length, top_k, n_size),
            (1, length, top_k, n_size),
        ],
        output_dtypes=[x.dtype, x.dtype],
    )


def exact_affine_moe_down(
    linear,
    x: mx.array,
    indices: mx.array,
    route_weights: mx.array,
    shared_linear,
    shared_x: mx.array,
):
    """Fuse affine routed/shared down projections and exact BF16 reduction."""
    if (
        not mx.metal.is_available()
        or not isinstance(linear, QuantizedSwitchLinear)
        or linear.mode != "affine"
        or linear.bits not in (4, 5)
        or linear.biases is None
        or "bias" in linear
        or x.ndim != 4
        or x.shape[0] != 1
        or not 1 < x.shape[1] <= 4
        or x.shape[2] != indices.shape[-1]
        or indices.shape != x.shape[:3]
        or route_weights.shape != indices.shape
        or route_weights.dtype != mx.float32
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
        or not isinstance(shared_linear, nn.QuantizedLinear)
        or shared_linear.mode != "affine"
        or shared_linear.bits != linear.bits
        or shared_linear.biases is None
        or "bias" in shared_linear
        or shared_x.shape != (1, x.shape[1], x.shape[-1])
        or shared_x.dtype != x.dtype
        or shared_linear.scales.dtype != x.dtype
        or shared_linear.biases.dtype != x.dtype
        or shared_linear.group_size != linear.group_size
    ):
        return None

    _, length, top_k, k_size = x.shape
    n_size = linear.weight.shape[1]
    if (
        k_size != linear.input_dims
        or k_size != shared_linear.weight.shape[1] * 32 // shared_linear.bits
        or shared_linear.weight.shape[0] != n_size
        or k_size % 512
        or n_size % 8
        or linear.group_size % 16
    ):
        return None

    x = mx.contiguous(x)
    indices = mx.contiguous(indices.astype(mx.int32))
    route_weights = mx.contiguous(route_weights)
    shared_x = mx.contiguous(shared_x)
    kernel = _affine_moe_down_kernel(
        linear.bits,
        x.dtype,
        length,
        k_size,
        n_size,
        top_k,
        linear.group_size,
    )
    return kernel(
        inputs=[
            x,
            indices,
            route_weights,
            linear.weight,
            linear.scales,
            linear.biases,
            shared_x,
            shared_linear.weight,
            shared_linear.scales,
            shared_linear.biases,
        ],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("N_SIZE", int(n_size)),
            ("TOP_K", int(top_k)),
            ("GROUP_SIZE", int(linear.group_size)),
        ],
        grid=(32, n_size // 2, length),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, length, n_size)],
        output_dtypes=[x.dtype],
    )[0]


def exact_hc_normalized_norm(connection, norm, x: mx.array):
    """Fuse HC input normalization, mixing, collapse, and output RMSNorm."""
    weight = connection.fn
    if (
        not mx.metal.is_available()
        or x.ndim != 4
        or x.dtype not in (mx.bfloat16, mx.float16)
        or x.shape[2:] != (4, 4096)
        or weight.ndim != 2
        or weight.shape != (24, 4 * 4096)
        or weight.dtype not in (mx.bfloat16, mx.float16)
        or norm.weight.dtype != x.dtype
    ):
        return None

    batch, length, hc_mult, width = x.shape
    kernel = _hc_normalized_norm_kernel(
        x.dtype,
        weight.dtype,
        hc_mult,
        width,
        connection.sinkhorn_iters,
        connection.hc_eps,
        norm.eps,
    )
    return kernel(
        inputs=[
            mx.contiguous(x),
            weight,
            connection.scale,
            connection.base,
            norm.weight,
        ],
        template=[
            ("T", x.dtype),
            ("W", weight.dtype),
            ("HC", int(hc_mult)),
            ("D", int(width)),
            ("ITERS", int(connection.sinkhorn_iters)),
            ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
            ("NORM_EPS_INT", round(norm.eps / 1e-9)),
        ],
        grid=(batch * length * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (batch, length, width),
            (batch, length, hc_mult),
            (batch, length, hc_mult, hc_mult),
        ],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


def exact_hc_norm(connection, norm, x: mx.array, mixes: mx.array):
    """Fuse HC collapse and its following RMSNorm with exact reductions."""
    if (
        not mx.metal.is_available()
        or x.ndim != 4
        or x.dtype not in (mx.bfloat16, mx.float16)
        or x.shape[2] != 4
        or x.shape[3] % 4
        or x.shape[3] > 4096
        or norm.weight.dtype != x.dtype
        or mixes.dtype != mx.float32
    ):
        return None

    batch, length, hc_mult, width = x.shape
    threads = max(32, ((width // 4 + 31) // 32) * 32)
    kernel = _hc_norm_kernel(
        x.dtype,
        hc_mult,
        width,
        connection.sinkhorn_iters,
        connection.hc_eps,
        norm.eps,
    )
    return kernel(
        inputs=[
            x,
            mixes,
            connection.scale,
            connection.base,
            norm.weight,
        ],
        template=[
            ("T", x.dtype),
            ("HC", int(hc_mult)),
            ("D", int(width)),
            ("ITERS", int(connection.sinkhorn_iters)),
            ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
            ("NORM_EPS_INT", round(norm.eps / 1e-9)),
        ],
        grid=(batch * length * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, length, width),
            (batch, length, hc_mult),
            (batch, length, hc_mult, hc_mult),
        ],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


def exact_fp32_decode_block_gemv(x: mx.array, weight: mx.array) -> Optional[mx.array]:
    """Share a BF16 weight across FP32 verifier rows using GEMV arithmetic."""
    if (
        not mx.metal.is_available()
        or x.ndim != 3
        or x.shape[0] != 1
        or not 1 < x.shape[1] <= 4
        or x.dtype != mx.float32
        or weight.ndim != 2
        or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or x.shape[-1] != weight.shape[-1]
        or x.shape[-1] != 4096
        or weight.shape[0] % 4
    ):
        return None

    _, length, k_size = x.shape
    out_size = weight.shape[0]
    kernel = _fp32_decode_block_gemv_kernel(weight.dtype, length, k_size, out_size)
    return kernel(
        inputs=[mx.contiguous(x), weight],
        template=[
            ("W", weight.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("OUT_SIZE", int(out_size)),
        ],
        grid=(32 * (out_size // 4), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, length, out_size)],
        output_dtypes=[mx.float32],
    )[0]


def exact_hc_expand(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
) -> Optional[mx.array]:
    """Expand verifier hyperconnections with decode-exact FMA ordering."""
    if (
        not mx.metal.is_available()
        or x.ndim != 3
        or residual.ndim != 4
        or residual.shape[:2] != x.shape[:2]
        or residual.shape[-1] != x.shape[-1]
        or residual.dtype != x.dtype
        or x.dtype not in (mx.bfloat16, mx.float16)
        or post.shape != residual.shape[:-1]
        or comb.shape != (*residual.shape[:-1], residual.shape[2])
        or post.dtype != mx.float32
        or comb.dtype != mx.float32
    ):
        return None

    batch, length, width = x.shape
    hc_mult = residual.shape[2]
    rows = batch * length
    if width % 8:
        return None
    simds = 8
    tiles = rows * (width // 8)
    kernel = _hc_expand_kernel(x.dtype, rows, hc_mult, width)
    output = kernel(
        inputs=[x, residual, post, comb],
        template=[
            ("T", x.dtype),
            ("ROWS", int(rows)),
            ("HC", int(hc_mult)),
            ("D", int(width)),
            ("SIMDS", simds),
        ],
        grid=(32 * ((tiles + simds - 1) // simds), simds, 1),
        threadgroup=(32, simds, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]
    return output


def exact_quantized_block_argmax(linear, x: mx.array) -> Optional[mx.array]:
    """Greedy affine projection without materializing vocabulary logits."""
    if (
        not mx.metal.is_available()
        or not isinstance(linear, nn.QuantizedLinear)
        or linear.mode != "affine"
        or linear.bits not in (4, 5)
        or linear.biases is None
        or "bias" in linear
        or x.ndim != 3
        or not 2 <= x.shape[0] <= 4
        or not 1 < x.shape[1] <= 4
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return None

    batch, length, k_size = x.shape
    n_size = linear.weight.shape[0]
    if k_size % linear.group_size or linear.group_size % 8 or n_size < 8 or n_size % 8:
        return None

    num_tiles = n_size // 8
    x = mx.contiguous(x)
    kernel = _affine_wide_argmax_kernel(
        linear.bits,
        x.dtype,
        batch,
        length,
        k_size,
        n_size,
        linear.group_size,
    )
    tile_values, tile_indices = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("BATCH_SIZE", int(batch)),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("N_SIZE", int(n_size)),
            ("GROUP_SIZE", int(linear.group_size)),
            ("NUM_TILES", int(num_tiles)),
        ],
        grid=(32, 2 * num_tiles, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[
            (batch, length, num_tiles),
            (batch, length, num_tiles),
        ],
        output_dtypes=[x.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(
        tile_indices,
        best_tile[..., None],
        axis=-1,
    ).squeeze(-1)


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_tree(item) for item in value]
    if type(value) is dict:
        return {key: _clone_cache_tree(item) for key, item in value.items()}
    return value


def _snapshot_single_cache(cache, incoming_tokens):
    if isinstance(cache, ArraysCache):
        # GLM's recurrent layer replaces its convolution and GDN arrays instead
        # of updating them in-place. Retaining those references avoids copying
        # roughly one full FP32 recurrent state per linear-attention layer.
        return (
            "arrays",
            list(cache.state),
            cache._left_padding,
            cache._left_padding_advance,
            cache._lengths,
            cache._lengths_advance,
        )
    if isinstance(cache, CacheList):
        return (
            "list",
            [_snapshot_single_cache(child, incoming_tokens) for child in cache.caches],
        )
    if isinstance(cache, (BatchKVCache, BatchQuantizedKVCache)):
        # Verification only appends to these buffers. Keep the original buffer
        # references and logical metadata; appended slots are outside ``_idx``
        # and are overwritten if rollback reuses them.
        return (
            "batch_append",
            cache.keys,
            cache.values,
            int(cache._idx),
            _clone_cache_tree(cache.offset),
            _clone_cache_tree(cache.left_padding),
            _clone_cache_tree(cache._right_padding),
        )
    if isinstance(cache, BatchPoolingCache):
        # Pool buffers are small (one compression window) but are updated
        # in-place. Preserve them while sharing the append-only pooled array.
        return (
            "batch_pooling",
            _clone_cache_tree(cache.buf_kv),
            _clone_cache_tree(cache.buf_gate),
            cache.pooled,
            list(cache.remainder),
            list(cache._pool_lengths),
            list(cache._lengths),
            list(cache._processed),
            list(cache.left_padding),
        )
    if isinstance(cache, PoolingCache):
        remainder = int(cache.remainder)
        total = remainder + int(incoming_tokens)
        overwrite = total % cache.ratio if total >= cache.ratio else total
        preserve = min(remainder, overwrite)
        buf_kv = (
            None
            if preserve == 0 or cache.buf_kv is None
            else _clone_cache_tree(cache.buf_kv[:, :preserve])
        )
        buf_gate = (
            None
            if preserve == 0 or cache.buf_gate is None
            else _clone_cache_tree(cache.buf_gate[:, :preserve])
        )
        return (
            "pooling",
            remainder,
            buf_kv,
            buf_gate,
            None if cache.pooled is None else cache.pooled.shape[1],
        )
    if isinstance(cache, KVCache):
        return ("append", cache.empty(), int(cache.offset))
    return (
        "full",
        _clone_cache_tree(cache.state),
        _clone_cache_tree(cache.meta_state),
    )


def _snapshot_cache(caches, incoming_tokens):
    return [_snapshot_single_cache(cache, incoming_tokens) for cache in caches]


def _restore_single_cache(cache, snapshot):
    kind = snapshot[0]
    if kind == "arrays":
        (
            _,
            state,
            left_padding,
            left_padding_advance,
            lengths,
            lengths_advance,
        ) = snapshot
        cache.state = list(state)
        cache._left_padding = left_padding
        cache._left_padding_advance = left_padding_advance
        cache._lengths = lengths
        cache._lengths_advance = lengths_advance
        return
    if kind == "list":
        for child, child_snapshot in zip(cache.caches, snapshot[1]):
            _restore_single_cache(child, child_snapshot)
        return
    if kind == "batch_append":
        (
            _,
            cache.keys,
            cache.values,
            cache._idx,
            cache.offset,
            cache.left_padding,
            cache._right_padding,
        ) = snapshot
        return
    if kind == "batch_pooling":
        (
            _,
            cache.buf_kv,
            cache.buf_gate,
            cache.pooled,
            remainder,
            pool_lengths,
            lengths,
            processed,
            left_padding,
        ) = snapshot
        cache.remainder = list(remainder)
        cache._pool_lengths = list(pool_lengths)
        cache._lengths = list(lengths)
        cache._processed = list(processed)
        cache.left_padding = list(left_padding)
        return
    if kind == "pooling":
        _, remainder, buf_kv, buf_gate, pooled_length = snapshot
        cache.remainder = remainder
        if buf_kv is not None:
            cache.buf_kv[:, : buf_kv.shape[1]] = buf_kv
            cache.buf_gate[:, : buf_gate.shape[1]] = buf_gate
        cache.pooled = (
            None if pooled_length is None else cache.pooled[:, :pooled_length]
        )
        return
    if kind == "append":
        _, was_empty, offset = snapshot
        if was_empty:
            cache.keys = cache.values = None
        cache.offset = offset
        return
    _, state, meta_state = snapshot
    cache.meta_state = _clone_cache_tree(meta_state)
    cache.state = _clone_cache_tree(state)


def _restore_cache(caches, snapshots):
    for cache, snapshot in zip(caches, snapshots):
        _restore_single_cache(cache, snapshot)


@mx.compile
def _combine_moe_outputs(routed, weights, shared):
    routed = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
    return routed + shared


@mx.compile
def _clamped_swiglu(gate, up, limit):
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


@mx.compile
def _scaled_rms_norm(inputs, scale, eps):
    return scale * mx.fast.rms_norm(inputs, None, eps)


class Glm5NextSpeculativeVerifier:
    """Run GLM-5-Next MTP verification and own its rollback state."""

    @staticmethod
    def _helpers():
        # language.py owns the shared sparse gather helper and imports this
        # verifier at module load time, so resolve it lazily.
        from . import language

        return language

    def logits_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        tied = language_model.args.tie_word_embeddings
        head = (
            language_model.model.embed_tokens.as_linear
            if tied
            else language_model.lm_head
        )
        if not tied:
            logits = exact_dense_block_linear(head, hidden)
            if logits is not None:
                return logits
        if not tied and hidden.ndim == 3 and hidden.shape[1] > 1:
            logits = exact_speculative_verify_weight(
                language_model.lm_head.weight, hidden
            )
            if logits is not None:
                return logits
        return self._block_linear(head, hidden)

    @staticmethod
    def _singleton_linear(linear, x: mx.array) -> mx.array:
        """Use Qwen's fused verifier projection with decode-exact arithmetic."""
        return _target_verify_linear(linear, x)

    @staticmethod
    def _batched_time_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
        """Project all verifier positions while preserving decode batch M.

        MLX selects its quantized kernel from the matrix row count. Flattening
        BxT changes that count and therefore its accumulation order. Treat T as
        an outer batch dimension instead: every position still runs with M=B,
        but the two decode-equivalent projections share one dispatch.
        """
        if not isinstance(linear, nn.QuantizedLinear) or x.ndim != 3 or x.shape[1] <= 1:
            return None

        batch, length, _ = x.shape
        transposed = mx.contiguous(x.transpose(1, 0, 2))
        weight = mx.broadcast_to(
            linear.weight[None],
            (length, *linear.weight.shape),
        )
        scales = mx.broadcast_to(
            linear.scales[None],
            (length, *linear.scales.shape),
        )
        biases = linear.get("biases")
        if biases is not None:
            biases = mx.broadcast_to(
                biases[None],
                (length, *biases.shape),
            )
        output = mx.quantized_matmul(
            transposed,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        ).transpose(1, 0, 2)
        if "bias" in linear:
            output = output + linear["bias"]
        return output

    def _block_linear(self, linear, x: mx.array) -> mx.array:
        """Fuse verifier time only where the projection remains decode exact."""
        if x.ndim != 3 or x.shape[1] <= 1:
            return linear(x)
        output = exact_dense_block_linear(linear, x)
        if output is not None:
            return output
        output = exact_quantized_block_linear(linear, x)
        if output is not None:
            return output
        if x.shape[0] == 1:
            return self._singleton_linear(linear, x)
        output = self._batched_time_quantized_linear(linear, x)
        if output is not None:
            return output
        return self._singleton_linear(linear, x)

    def _dense_mlp(self, mlp, x):
        gate, up = mx.split(self._block_linear(mlp.gate_up_proj, x), 2, axis=-1)
        hidden = _clamped_swiglu(gate, up, mlp.swiglu_limit)
        return self._block_linear(mlp.down_proj, hidden)

    def _moe(self, moe, x):
        if x.shape[0] > 1:
            # Keeping verifier time as the matrix-row dimension lets MLX reuse
            # every selected/shared expert dispatch. For B2-B4 this follows
            # the same quantized accumulation path as decode and is exact.
            return moe(x)

        # Routing and the shared expert must keep the decode Bx1 shape. The
        # selected-expert gather-QMVs, however, are row independent and remain
        # bit-exact when verifier time is flattened into their row dimension.
        # Flatten only that dominant path so one set of launches handles the
        # complete verifier block without changing greedy output.
        # The FP32 gate produces identical routes and weights at B=1 when the
        # two verifier positions share one call.
        batch, length, width = x.shape
        gate = moe.gate
        fp32_x = x.astype(mx.float32)
        logits = exact_fp32_decode_block_gemv(fp32_x, gate.weight)
        if logits is None:
            gate_parts = []
            for index in range(length):
                part = gate(mx.contiguous(x[:, index : index + 1]))
                mx.async_eval(*part)
                gate_parts.append(part)
            indices = mx.concatenate([part[0] for part in gate_parts], axis=1)
            weights = mx.concatenate([part[1] for part in gate_parts], axis=1)
        else:
            indices, weights = self._helpers()._expert_select(
                logits,
                gate.e_score_correction_bias,
                gate.top_k,
                gate.n_group,
                gate.topk_group,
                gate.routed_scaling_factor,
                gate.norm_topk_prob,
            )
        top_k = indices.shape[-1]
        flat_x = mx.contiguous(x).reshape(batch * length, width)
        flat_indices = indices.reshape(batch * length, top_k)
        switch = moe.switch_mlp
        gate_up = exact_affine_switch_gate_up(switch, x, indices)
        if gate_up is None:
            projected = mx.expand_dims(flat_x, (-2, -3))
            up = switch.up_proj(projected, flat_indices, sorted_indices=False)
            gate = switch.gate_proj(projected, flat_indices, sorted_indices=False)
        else:
            up, gate = gate_up
        activated = switch.activation(up, gate)
        shared_gate, shared_up = mx.split(
            self._block_linear(moe.shared_experts.gate_up_proj, x),
            2,
            axis=-1,
        )
        shared_hidden = _clamped_swiglu(
            shared_gate,
            shared_up,
            moe.shared_experts.swiglu_limit,
        )
        # Affine-4/5 verification can consume the routed and shared expert
        # activations together.  Prefer that path before materializing the
        # shared down projection; it removes one full 4096-wide quantized
        # dispatch per MoE layer.
        fused_moe = (
            exact_affine_moe_down(
                switch.down_proj,
                activated,
                indices,
                weights,
                moe.shared_experts.down_proj,
                shared_hidden,
            )
            if gate_up is not None
            else None
        )
        if fused_moe is not None:
            return fused_moe

        shared = self._block_linear(
            moe.shared_experts.down_proj,
            shared_hidden,
        )
        if gate_up is not None:
            activated = activated.reshape(batch * length, top_k, 1, -1)
        routed = switch.down_proj(
            activated,
            flat_indices,
            sorted_indices=False,
        )
        routed = routed.squeeze(-2).reshape(batch, length, top_k, -1)
        return _combine_moe_outputs(routed, weights, shared)

    def _linear_attention(self, attention, inputs, mask, cache):
        updates = []
        output = attention(
            inputs,
            mask,
            cache,
            rollback_sink=updates,
            linear_fn=self._block_linear,
            output_linear_fn=self._block_linear,
            timewise_fn=self._timewise,
            output_gate_fn=self._linear_output_gate_timewise,
            scaled_norm_fn=_scaled_rms_norm,
        )
        return output, updates[0][:12]

    @staticmethod
    def _linear_output_gate_timewise(norm, output, gate):
        # RMSNorm reduces each head row independently, so verifier time does
        # not alter its accumulation tree. Keeping BxT intact lets the compiled
        # output norm, sigmoid, and gate run as one graph instead of two graphs
        # plus a concatenate for every speculative position.
        result = norm(output) * mx.sigmoid(gate)
        mx.async_eval(result)
        return result

    @staticmethod
    def _merge_sparse_updates(updates):
        merged = {
            "new_latent": mx.concatenate(
                [update["new_latent"] for update in updates],
                axis=2,
            )
        }
        if "indexer" not in updates[0]:
            return merged
        merged.update(
            indexer=updates[0]["indexer"],
            index_keys=mx.concatenate(
                [update["index_keys"] for update in updates], axis=1
            ),
            index_gates=mx.concatenate(
                [update["index_gates"] for update in updates], axis=1
            ),
            query_valid=mx.concatenate(
                [update["query_valid"] for update in updates], axis=1
            ),
            cache_offset=updates[0]["cache_offset"],
        )
        return merged

    def _sparse_attention(
        self,
        attention,
        inputs,
        mask,
        cache,
        prev_topk_indices,
    ):
        batch, length, _ = inputs.shape
        q_a, kv_a = mx.split(
            self._block_linear(attention.qkv_a_proj, inputs),
            (attention.q_lora_rank,),
            axis=-1,
        )
        q_resid = self._timewise(attention.q_a_layernorm, q_a)
        q = (
            self._block_linear(attention.q_b_proj, q_resid)
            .reshape(batch, length, attention.num_heads, attention.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = self._timewise(attention.kv_a_layernorm, kv_a)[:, None]
        index_projected = None
        if attention.indexer is not None:
            indexer = attention.indexer
            index_keys = self._timewise(
                indexer.k_norm,
                self._block_linear(indexer.wk, inputs),
            )
            index_gates = exact_fp32_decode_block_gemv(
                inputs.astype(mx.float32),
                indexer.index_kpool_compress_gate,
            )
            if index_gates is None:
                index_gates = mx.concatenate(
                    [
                        inputs[:, index : index + 1].astype(mx.float32)
                        @ indexer.index_kpool_compress_gate.T
                        for index in range(length)
                    ],
                    axis=1,
                )

            # The scorer projections are skipped while all historical pools
            # fit in the exact candidate set. Once the history exceeds top-k,
            # hoist them across verifier time just like the key projection.
            score_pool_limit = indexer.index_topk
            cache_offset = 0 if cache is None else cache[0].offset
            needs_score_projection = isinstance(cache_offset, mx.array) or (
                cache_offset + length > score_pool_limit
            )
            if needs_score_projection:
                index_q = self._block_linear(indexer.wq_b, q_resid).reshape(
                    batch,
                    length,
                    indexer.n_heads,
                    indexer.head_dim,
                )
                index_weights = (
                    self._block_linear(indexer.weights_proj, inputs).astype(mx.float32)
                    * indexer.n_heads**-0.5
                )
            else:
                index_q = index_weights = None
            index_projected = (
                index_keys,
                index_gates,
                index_q,
                index_weights,
            )
        outputs = []
        topks = []
        updates = []
        for index in range(inputs.shape[1]):
            step_updates = []
            output, topk = self._attention(
                attention,
                mx.contiguous(inputs[:, index : index + 1]),
                None if mask is None else mask[:, index : index + 1],
                cache,
                (
                    None
                    if prev_topk_indices is None
                    else prev_topk_indices[:, index : index + 1]
                ),
                step_updates,
                projected=(
                    q_resid[:, index : index + 1],
                    q[:, :, index : index + 1],
                    new_latent[:, :, index : index + 1],
                    (
                        None
                        if index_projected is None
                        else tuple(
                            None if value is None else value[:, index : index + 1]
                            for value in index_projected
                        )
                    ),
                ),
                project_output=False,
            )
            mx.async_eval(output, topk)
            outputs.append(output)
            topks.append(topk)
            updates.append(step_updates[0])
        return (
            self._block_linear(
                attention.o_proj,
                mx.concatenate(outputs, axis=1),
            ),
            mx.concatenate(topks, axis=1),
            self._merge_sparse_updates(updates),
        )

    @staticmethod
    def _timewise(fn, x: mx.array) -> mx.array:
        return fn(x)

    @staticmethod
    def _head_timewise(fn, x: mx.array) -> mx.array:
        if x.ndim != 4 or x.shape[2] <= 1:
            return fn(x)
        outputs = []
        for index in range(x.shape[2]):
            output = fn(mx.contiguous(x[:, :, index : index + 1]))
            mx.async_eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=2)

    @staticmethod
    def _hc_expand_timewise(x, residual, post, comb):
        output = exact_hc_expand(x, residual, post, comb)
        if output is not None:
            return output
        return hc_expand(x, residual, post, comb)

    @staticmethod
    def _hc_inputs(connection, x):
        y = x.astype(mx.float32)
        if x.shape[1] <= 1 or x.shape[0] > 1:
            normalized = mx.fast.rms_norm(
                y.flatten(-2),
                None,
                connection.norm_eps,
            )
            return y, normalized @ connection.fn.T

        # At B=1, flattening verifier time changes MLX's FP32 matmul reduction
        # order. Keep each mix projection decode-shaped while sharing the
        # Sinkhorn/collapse work across the complete verifier block.
        mixes = []
        normalized_steps = []
        for index in range(x.shape[1]):
            normalized = mx.fast.rms_norm(
                y[:, index : index + 1].flatten(-2),
                None,
                connection.norm_eps,
            )
            normalized_steps.append(normalized)
        for normalized in normalized_steps:
            mixes.append(normalized @ connection.fn.T)
        return y, mx.concatenate(mixes, axis=1)

    def _hc(self, connection, x):
        if _hc_kernel is None:
            return self._timewise(connection, x)
        y, mixes = self._hc_inputs(connection, x)
        return _hc_kernel(
            x,
            y,
            mixes,
            connection.scale,
            connection.base,
            connection.hc_mult,
            connection.sinkhorn_iters,
            connection.hc_eps,
        )

    def _hc_norm(self, connection, norm, x):
        # The fixed GLM verifier shape can keep the complete HC ingress in one
        # dispatch: input RMSNorm, mix projection, Sinkhorn/collapse, and the
        # decoder RMSNorm.  Besides avoiding three intermediate arrays, this
        # preserves the decode reduction tree used by the split exact path.
        # The single-row Metal kernel is decode exact at B=1. At B>1 its
        # FP32 post/comb rounding can differ from the target's batched HC path
        # by one ULP, which is enough to flip a later greedy token. Keep the
        # exact split mix/collapse kernel below for batched verification.
        output = (
            exact_hc_normalized_norm(connection, norm, x) if x.shape[0] == 1 else None
        )
        if output is not None:
            return output
        if _hc_kernel is not None:
            _y, mixes = self._hc_inputs(connection, x)
            output = self._hc_norm_from_mixes(connection, norm, x, mixes)
            if output is not None:
                return output
        collapsed, post, comb = self._hc(connection, x)
        return self._timewise(norm, collapsed), post, comb

    @staticmethod
    def _hc_norm_from_mixes(connection, norm, x, mixes):
        return exact_hc_norm(connection, norm, x, mixes)

    def argmax_from_hidden(self, language_model, hidden: mx.array) -> mx.array:
        if not language_model.args.tie_word_embeddings:
            head = language_model.lm_head
            output = exact_quantized_block_argmax(head, hidden)
            if output is not None:
                return output
            if hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[1] > 1:
                output = _target_verify_quantized_argmax(head, hidden)
                if output is not None:
                    return output
        return mx.argmax(self.logits_from_hidden(language_model, hidden), axis=-1)

    def _attention(
        self,
        attention,
        x,
        padding_mask,
        cache,
        prev_topk_indices,
        rollback_sink,
        projected=None,
        project_output=True,
    ):
        batch, length, _ = x.shape
        if projected is None:
            q_a, kv_a = mx.split(
                self._block_linear(attention.qkv_a_proj, x),
                (attention.q_lora_rank,),
                axis=-1,
            )
            q_resid = self._timewise(attention.q_a_layernorm, q_a)
            q = (
                self._block_linear(attention.q_b_proj, q_resid)
                .reshape(batch, length, attention.num_heads, attention.q_head_dim)
                .transpose(0, 2, 1, 3)
            )
            new_latent = self._timewise(attention.kv_a_layernorm, kv_a)[:, None]
            index_projected = None
        else:
            q_resid, q, new_latent, index_projected = projected

        if cache is None:
            kv_cache = index_cache = pool_cache = None
            latent = new_latent
            cache_offset = 0
        else:
            kv_cache = cache[0]
            cache_offset = kv_cache.offset
            if attention.indexer is None:
                index_cache = pool_cache = None
            else:
                index_cache = cache[1]
                pool_cache = cache[2]
            latent, _ = kv_cache.update_and_fetch(
                new_latent,
                mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype),
            )

        cache_update = {"new_latent": new_latent}
        if attention.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared indexer layer has no previous top-k indices.")
            topk = prev_topk_indices
        elif length > 1:
            topk_parts = []
            index_updates = []
            for index in range(length):
                index_update = {}
                topk_parts.append(
                    attention.indexer(
                        x[:, index : index + 1],
                        q_resid[:, index : index + 1],
                        (
                            None
                            if padding_mask is None
                            else padding_mask[:, index : index + 1]
                        ),
                        index_cache,
                        pool_cache,
                        cache_offset + index,
                        cache_update_sink=index_update,
                        linear_fn=self._singleton_linear,
                    )
                )
                index_updates.append(index_update)
            topk = mx.concatenate(topk_parts, axis=1)
            cache_update.update(
                indexer=attention.indexer,
                index_keys=mx.concatenate(
                    [update["index_keys"] for update in index_updates], axis=1
                ),
                index_gates=mx.concatenate(
                    [update["index_gates"] for update in index_updates], axis=1
                ),
                query_valid=mx.concatenate(
                    [update["query_valid"] for update in index_updates], axis=1
                ),
                cache_offset=cache_offset,
            )
        else:
            topk = attention.indexer(
                x,
                q_resid,
                padding_mask,
                index_cache,
                pool_cache,
                cache_offset,
                cache_update_sink=cache_update,
                linear_fn=self._singleton_linear,
                projected=index_projected,
            )
        rollback_sink.append(cache_update)

        kv_length = latent.shape[2]
        valid = (topk >= 0) & (topk < kv_length)
        safe = mx.clip(topk, 0, max(kv_length - 1, 0))
        if length == 1:
            selected = mx.take_along_axis(
                latent,
                safe[:, None, 0, :, None],
                axis=2,
            )
            q = self._head_timewise(attention.embed_q, q)
            out = scaled_dot_product_attention(
                q,
                selected,
                selected,
                cache=kv_cache,
                scale=attention.scale,
                mask=valid[:, None],
            )
            out = self._head_timewise(attention.unembed_out, out)
        else:
            # Retain latent-MLA decode arithmetic. Reprojecting the complete
            # latent cache is algebraically equivalent but changes MXFP8
            # rounding and can flip a close greedy target token.
            selected = self._helpers()._sparse_head_gather(latent, safe)
            latent_q = self._head_timewise(attention.embed_q, q)
            folded_q = latent_q.transpose(0, 2, 1, 3).reshape(
                batch * length,
                attention.num_heads,
                1,
                attention.kv_lora_rank,
            )
            folded_mask = valid.reshape(batch * length, 1, 1, topk.shape[-1])
            out = scaled_dot_product_attention(
                folded_q,
                selected,
                selected,
                cache=None,
                scale=attention.scale,
                mask=folded_mask,
            )
            out = out.reshape(
                batch, length, attention.num_heads, attention.kv_lora_rank
            )
            out = out.transpose(0, 2, 1, 3)
            out = self._head_timewise(attention.unembed_out, out)

        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if project_output:
            out = self._block_linear(attention.o_proj, out)
        return out, topk

    def _layer(
        self,
        layer,
        hidden,
        mask,
        cache,
        prev_topk_indices,
        rollback_sink,
    ):
        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.attn_hc,
            layer.input_layernorm,
            hidden,
        )
        if layer.block_type == "linear_attention":
            collapsed, linear_update = self._linear_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
            )
            rollback_sink.append(linear_update)
            topk = prev_topk_indices
        else:
            collapsed, topk, sparse_update = self._sparse_attention(
                layer.self_attn,
                collapsed,
                mask,
                cache,
                prev_topk_indices,
            )
            rollback_sink.append(sparse_update)
        hidden = self._hc_expand_timewise(collapsed, residual, post, comb)

        residual = hidden
        collapsed, post, comb = self._hc_norm(
            layer.ffn_hc,
            layer.post_attention_layernorm,
            hidden,
        )
        if hasattr(layer.mlp, "switch_mlp"):
            collapsed = self._moe(layer.mlp, collapsed)
        else:
            collapsed = self._dense_mlp(layer.mlp, collapsed)
        return self._hc_expand_timewise(collapsed, residual, post, comb), topk

    def _model(
        self,
        model,
        inputs,
        inputs_embeds,
        cache,
        attention_mask,
        hidden_sink,
        rollback_sink,
    ):
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(model.layers)
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mx.bool_)
            if attention_mask.shape[-1] != hidden.shape[1]:
                attention_mask = attention_mask[..., -hidden.shape[1] :]

        hidden = mx.repeat(hidden[:, :, None], model.config.hc_mult, axis=2)
        topk = None
        for layer, layer_cache in zip(model.layers, cache):
            layer_mask = attention_mask
            if layer.block_type == "linear_attention" and layer_cache is not None:
                layer_mask = create_ssm_mask(hidden[:, :, 0], layer_cache)
            hidden, topk = self._layer(
                layer,
                hidden,
                layer_mask,
                layer_cache,
                topk,
                rollback_sink,
            )

        hidden = self._timewise(model.norm, hidden.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(hidden)
        return hidden

    def __call__(
        self,
        language_model,
        inputs,
        *,
        inputs_embeds=None,
        cache=None,
        attention_mask=None,
        hidden_sink=None,
        rollback_sink=None,
        return_shared_kv=False,
        skip_logits=False,
    ) -> LanguageModelOutput:
        if rollback_sink is None:
            rollback_sink = []
        hidden = self._model(
            language_model.model,
            inputs,
            inputs_embeds,
            cache,
            attention_mask,
            hidden_sink,
            rollback_sink,
        )
        logits = (
            None if skip_logits else self.logits_from_hidden(language_model, hidden)
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def verify(
        self,
        language_model,
        inputs,
        cache,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
    ):
        cache_snapshot = _snapshot_cache(cache, inputs.shape[1])
        hidden_sink = []
        rollback_updates = []
        output = self(
            language_model,
            inputs,
            cache=cache,
            hidden_sink=hidden_sink,
            rollback_sink=rollback_updates,
            return_shared_kv=True,
            skip_logits=True,
        )
        hidden = output.hidden_states[-1]
        rollback_state = (cache_snapshot, rollback_updates)
        if sampler is None:
            return hidden, {}, rollback_state
        return (
            hidden,
            {},
            rollback_state,
            sampler(self.logits_from_hidden(language_model, hidden)),
        )

    @staticmethod
    def _prepare_cache(cache, valid_lengths, right_padding):
        prepare = getattr(cache, "prepare", None)
        if callable(prepare):
            prepare(lengths=valid_lengths, right_padding=right_padding)

    @staticmethod
    def _finalize_cache(cache):
        finalize = getattr(cache, "finalize", None)
        if callable(finalize):
            finalize()

    def _replay_attention_cache(self, cache, update, valid_lengths, keep):
        if cache is None:
            return

        batch = len(valid_lengths)
        ragged = len(set(valid_lengths)) > 1
        right_padding = [keep - length for length in valid_lengths] if ragged else None
        kv_cache = cache[0]
        touched = [kv_cache]

        indexer = update.get("indexer")
        if indexer is not None:
            index_cache = cache[1]
            pool_cache = cache[2]
            touched.extend([index_cache, pool_cache])

        if ragged:
            for entry in touched:
                self._prepare_cache(entry, valid_lengths, right_padding)

        new_latent = update["new_latent"][:, :, :keep]
        kv_cache.update_and_fetch(
            new_latent,
            mx.zeros((batch, 1, keep, 0), dtype=new_latent.dtype),
        )

        if indexer is not None:
            valid = (
                mx.arange(keep)[None] < mx.array(valid_lengths, dtype=mx.int32)[:, None]
            )
            query_valid = update["query_valid"][:, :keep] & valid
            keys = update["index_keys"][:, :keep]
            gates = update["index_gates"][:, :keep]
            for index in range(keep):
                index_cache.update_and_fetch(
                    query_valid[:, None, index : index + 1, None],
                    mx.zeros((batch, 1, 1, 0), dtype=mx.bool_),
                )

                ready_keys, ready_gates, _ = pool_cache.accumulate_windows(
                    keys[:, index : index + 1],
                    gates[:, index : index + 1],
                    update["cache_offset"] + index,
                )
                new_pool_keys = indexer._compress_pools(ready_keys, ready_gates)
                pool_cache.update_and_fetch(new_pool_keys)

        if ragged:
            for entry in touched:
                self._finalize_cache(entry)

    @staticmethod
    def _initial_linear_state(update):
        q, _k, v, *_rest = update
        initial_state = update[7]
        if initial_state is not None:
            return initial_state
        batch = q.shape[0]
        heads, value_dim = v.shape[-2:]
        key_dim = q.shape[-1]
        return mx.zeros(
            (batch, heads, value_dim, key_dim),
            dtype=mx.float32,
        )

    def _replay_linear_caches(self, entries, valid_lengths, keep):
        if not entries:
            return

        if all(len(update) > 13 for _cache, update in entries):
            indices = mx.array(valid_lengths, dtype=mx.int32) - 1
            for cache, update in entries:
                conv_states = update[12]
                ssm_states = update[13]
                conv_indices = indices.reshape(
                    indices.shape[0], 1, *([1] * (conv_states.ndim - 2))
                )
                ssm_indices = indices.reshape(
                    indices.shape[0], 1, *([1] * (ssm_states.ndim - 2))
                )
                cache[0] = mx.take_along_axis(
                    conv_states,
                    conv_indices,
                    axis=1,
                ).squeeze(1)
                cache[1] = cache[2] = None
                cache[3] = mx.take_along_axis(
                    ssm_states,
                    ssm_indices,
                    axis=1,
                ).squeeze(1)
                self._finalize_cache(cache)
            return

        batch = len(valid_lengths)
        valid = mx.arange(keep)[None] < mx.array(valid_lengths, dtype=mx.int32)[:, None]
        q_parts, k_parts, v_parts, a_parts, b_parts = [], [], [], [], []
        A_log_parts, dt_bias_parts, state_parts, mask_parts = [], [], [], []
        lower_bounds = []

        for _cache, update in entries:
            q, k, v, a, b, A_log, dt_bias, *_ = update
            q_parts.append(q[:, :keep])
            k_parts.append(k[:, :keep])
            v_parts.append(v[:, :keep])
            a_parts.append(a[:, :keep])
            b_parts.append(b[:, :keep])
            A_log_parts.append(
                mx.broadcast_to(
                    A_log[None, None],
                    (batch, 1, *A_log.shape),
                )
            )
            dt_bias_parts.append(
                mx.broadcast_to(
                    dt_bias[None, None],
                    (batch, 1, *dt_bias.shape),
                )
            )
            state_parts.append(self._initial_linear_state(update))
            layer_mask = update[8]
            if layer_mask is None:
                layer_mask = valid
            else:
                layer_mask = layer_mask[:, :keep] & valid
            mask_parts.append(layer_mask)
            lower_bounds.append(update[11])

        if len(set(lower_bounds)) != 1:
            raise ValueError("GLM Gated Delta layers must share a lower bound.")

        _output, states = gated_delta_update(
            mx.concatenate(q_parts, axis=0),
            mx.concatenate(k_parts, axis=0),
            mx.concatenate(v_parts, axis=0),
            mx.concatenate(a_parts, axis=0),
            mx.concatenate(b_parts, axis=0),
            mx.concatenate(A_log_parts, axis=0),
            mx.concatenate(dt_bias_parts, axis=0),
            state=mx.concatenate(state_parts, axis=0),
            mask=mx.concatenate(mask_parts, axis=0),
            use_kernel=True,
            lower_bound=lower_bounds[0],
        )

        offset = 0
        for cache, update in entries:
            conv_input = update[9]
            kernel_size = int(update[10])
            cache[0] = mx.concatenate(
                [
                    conv_input[
                        row : row + 1,
                        length : length + kernel_size - 1,
                    ]
                    for row, length in enumerate(valid_lengths)
                ],
                axis=0,
            )
            cache[1] = cache[2] = None
            cache[3] = states[offset : offset + batch]
            offset += batch
            self._finalize_cache(cache)

    def rollback(
        self,
        language_model,
        caches: List[Any],
        rollback_state,
        accepted,
        block_size: int,
    ) -> int:
        del block_size
        if isinstance(accepted, int):
            accepted_values = [accepted]
        elif isinstance(accepted, mx.array):
            accepted_values = [int(value) for value in accepted.tolist()]
        else:
            accepted_values = [int(value) for value in accepted]

        cache_snapshot, rollback_updates = rollback_state
        _restore_cache(caches, cache_snapshot)
        valid_lengths = [value + 1 for value in accepted_values]
        keep = max(valid_lengths, default=0)
        if not keep:
            return 0

        linear_entries = []
        for layer, cache, update in zip(
            language_model.model.layers,
            caches,
            rollback_updates,
        ):
            if layer.block_type == "linear_attention":
                linear_entries.append((cache, update))
            else:
                self._replay_attention_cache(cache, update, valid_lengths, keep)
        self._replay_linear_caches(linear_entries, valid_lengths, keep)
        return max(accepted_values)


__all__ = ["Glm5NextSpeculativeVerifier"]
