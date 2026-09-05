from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

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
int batch = token / VERIFY_T;
int time = token - batch * VERIFY_T;
int batch_route = batch * VERIFY_T * TOP_K;
int expert = int(indices[route]);
int paired_route = -1;
if ((time & 1) == 0 && time + 1 < VERIFY_T) {
  int next_route = batch_route + (time + 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[next_route + other]) == expert) {
      paired_route = next_route + other;
      break;
    }
  }
} else if ((time & 1) != 0) {
  int previous_route = batch_route + (time - 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[previous_route + other]) == expert) {
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
int paired_token = paired_route >= 0 ? paired_route / TOP_K : token;
xk[1] = x + paired_token * K_SIZE +
    int(simd_lid) * VALUES_PER_THREAD;

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

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  if (simd_lid == 0 && n < N_SIZE) {
    y[int(token) * N_SIZE + n] =
        T(routed_sum[row] + shared_y[int(token) * N_SIZE + n]);
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
        header=_affine_exact_header(bits, group_size, 4, 2),
        source=_affine_source(_AFFINE5_SWITCH_GATE_UP_SOURCE, bits),
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
            "shared_y",
        ],
        output_names=["y"],
        header=_affine_exact_header(bits, group_size, 2, 4),
        source=_affine_source(_AFFINE5_MOE_DOWN_SOURCE, bits),
    )


def exact_dense_block_linear(linear, x: mx.array) -> Optional[mx.array]:
    """Project every verifier position with the target's native batch shape."""
    if (
        not isinstance(linear, nn.Linear)
        or "bias" in linear
        or x.ndim != 3
        or x.shape[0] < 1
        or x.shape[1] <= 1
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.weight.dtype != x.dtype
        or linear.weight.shape[1] != x.shape[-1]
    ):
        return None

    return mx.concatenate(
        [
            linear(mx.contiguous(x[:, position : position + 1]))
            for position in range(x.shape[1])
        ],
        axis=1,
    )


def exact_affine_switch_gate_up(switch, x: mx.array, indices: mx.array):
    """Project selected affine up/gate weights in one verifier dispatch."""
    up = getattr(switch, "up_proj", None)
    gate = getattr(switch, "gate_proj", None)
    linears = (up, gate)
    if (
        not mx.metal.is_available()
        or x.ndim != 3
        or x.shape[0] < 1
        or x.shape[1] <= 1
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

    batch, length, k_size = x.shape
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
    if batch > 1:
        projected = mx.expand_dims(x, (-2, -3))
        return (
            up(projected, indices, sorted_indices=False).squeeze(-2),
            gate(projected, indices, sorted_indices=False).squeeze(-2),
        )

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
        grid=(32, 2 * (n_size // 8), batch * length * top_k),
        threadgroup=(32, 2, 1),
        output_shapes=[
            (batch, length, top_k, n_size),
            (batch, length, top_k, n_size),
        ],
        output_dtypes=[x.dtype, x.dtype],
    )


def exact_affine_moe_down(
    linear,
    x: mx.array,
    indices: mx.array,
    route_weights: mx.array,
    shared: mx.array,
):
    """Fuse affine routed down projection, weighting, and shared addition."""
    if (
        not mx.metal.is_available()
        or not isinstance(linear, QuantizedSwitchLinear)
        or linear.mode != "affine"
        or linear.bits not in (4, 5)
        or linear.biases is None
        or "bias" in linear
        or x.ndim != 4
        or x.shape[0] < 1
        or x.shape[1] <= 1
        or x.shape[2] != indices.shape[-1]
        or indices.shape != x.shape[:3]
        or route_weights.shape != indices.shape
        or route_weights.dtype != mx.float32
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
        or shared.shape != (x.shape[0], x.shape[1], linear.weight.shape[1])
        or shared.dtype != x.dtype
    ):
        return None

    batch, length, top_k, k_size = x.shape
    n_size = linear.weight.shape[1]
    if (
        k_size != linear.input_dims
        or k_size % 512
        or n_size % 8
        or linear.group_size % 16
    ):
        return None

    x = mx.contiguous(x)
    indices = mx.contiguous(indices.astype(mx.int32))
    route_weights = mx.contiguous(route_weights)
    shared = mx.contiguous(shared)
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
            shared,
        ],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(k_size)),
            ("N_SIZE", int(n_size)),
            ("TOP_K", int(top_k)),
            ("GROUP_SIZE", int(linear.group_size)),
        ],
        grid=(32, n_size // 2, batch * length),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, length, n_size)],
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
    """Project FP32 verifier positions with the target's native batch shape."""
    if (
        x.ndim != 3
        or x.shape[0] < 1
        or x.shape[1] <= 1
        or x.dtype != mx.float32
        or weight.ndim != 2
        or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or x.shape[-1] != weight.shape[-1]
    ):
        return None

    return mx.concatenate(
        [
            mx.matmul(
                mx.contiguous(x[:, position : position + 1]),
                weight.T,
            )
            for position in range(x.shape[1])
        ],
        axis=1,
    )


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


@mx.compile
def combine_moe_outputs(routed, weights, shared):
    routed = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
    return routed + shared


@mx.compile
def clamped_swiglu(gate, up, limit):
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


@mx.compile
def scaled_rms_norm(inputs, scale, eps):
    return scale * mx.fast.rms_norm(inputs, None, eps)
