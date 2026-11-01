"""Masked NVFP4 decode matvec, adapted from MLX's fp_qmv_fast_impl.

MLX copyright Apple Inc., MIT license. Same 16-value dot grouping, FP32
accumulation, global scale and output rounding; skip unowned route tiles.
"""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _masked_qmv():
    return mx.fast.metal_kernel(
        name="ml4_masked_native_nvfp4_qmv",
        input_names=["x", "w", "s", "g", "indices"],
        output_names=["out"],
        header=_HEADER,
        source=r"""
        uint route = threadgroup_position_in_grid.z;
        uint lane = thread_index_in_simdgroup;
        uint row0 = threadgroup_position_in_grid.y * 8
                  + simdgroup_index_in_threadgroup * 4;
        int expert = int(indices[route]);
        if (expert < 0) {
            if (lane == 0) {
                for (uint r = 0; r < 4; ++r)
                    if (row0+r < N) out[route*N+row0+r] = T(0);
            }
            return;
        }
        const device T* xp = x + (route / ROUTES_PER_X) * K;
        const device uint* wp = w + size_t(expert) * N * (K/8);
        const device uchar* sp = s + size_t(expert) * N * (K/16);
        fp_qmv_fast_impl<T,16,4,true,false>(
            wp, sp, g + expert, xp, out + route*N,
            x_shape[x_ndim-1], w_shape[1],
            uint3(0, threadgroup_position_in_grid.y, 0),
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """,
        compile_options={"math_mode": "safe"},
    )


def supports_masked_qmv(linear):
    """Only the native NVFP4 vector layout and scaling tested by this kernel."""
    from ...quantization.nvfp4 import ScaledQuantizedSwitchLinear
    from ..switch_layers import QuantizedSwitchLinear

    if not isinstance(linear, (QuantizedSwitchLinear, ScaledQuantizedSwitchLinear)):
        return False
    if (
        linear.mode != "nvfp4"
        or linear.bits != 4
        or linear.group_size != 16
        or linear.input_dims % 512
        or linear.output_dims % 8
        or linear.weight.dtype != mx.uint32
        or linear.scales.dtype != mx.uint8
        or "bias" in linear
        or "pre_quant_scale" in linear
        or linear.get("biases") is not None
        or getattr(linear, "_quantize_activations", False)
    ):
        return False
    if isinstance(linear, ScaledQuantizedSwitchLinear):
        return linear.global_scale.size in (1, linear.num_experts)
    return True


def masked_qmv(x, linear, indices):
    k, n = linear.input_dims, linear.output_dims
    assert k % 512 == 0 and n % 8 == 0
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
        # Small custom-kernel inputs use Metal's constant address space.
        factor = mx.concatenate([factor, mx.zeros((64 - e,), mx.float32)])
    per_x = indices.size // (x.size // k)
    return _masked_qmv()(
        inputs=[x, linear.weight, linear.scales, factor, indices],
        template=[("T", x.dtype), ("K", k), ("N", n), ("ROUTES_PER_X", per_x)],
        grid=(32, ((n + 7) // 8) * 2, indices.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


# Keep the native runtime loop bounds and arithmetic order: constexpr loop
# bounds or fast math can change BF16 rounding and subsequent expert routing.
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
mlx/backend/metal/kernels/{fp4.h,fp8.h,fp_quantized.h}.
*/
// Copyright Apple Inc. MIT license. From MLX 1ba7f969.


constant constexpr float F8E4M3_MAX = 448.0f;
constant constexpr float F4E2M1_MAX = 6.0f;

struct fp4_e2m1 {
  fp4_e2m1(float x) thread {
    if (metal::isnan(x)) {
      bits = 0x7;
      return;
    }

    const uint8_t sign_bit = (metal::signbit(x)) ? 0x8 : 0x0;
    x = metal::abs(x);

    if (x > 5.0f) {
      bits = 0x7;
    } else if (x >= 3.5f) {
      bits = 0x6;
    } else if (x > 2.5f) {
      bits = 0x5;
    } else if (x >= 1.75f) {
      bits = 0x4;
    } else if (x > 1.25f) {
      bits = 0x3;
    } else if (x >= 0.75f) {
      bits = 0x2;
    } else if (x > 0.25f) {
      bits = 0x1;
    } else {
      bits = 0x0;
    }
    bits |= sign_bit;
  }

  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  uint8_t bits;
};



struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) thread {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      bits = (f_bits > 0x7F800000) ? 0x7F : 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) thread {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator bfloat16_t() thread {
    uint16_t out = (bits == 0)
        ? 0x40
        : (bits == 0xFF ? 0x7FC0 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() thread {
    uint32_t out = (bits == 0)
        ? 0x400000
        : (bits == 0xFF ? 0x7FC00000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};

// Smallest E8M0 >= x. Scales are amax/max_element, so rounding one down
// leaves the block's largest elements outside the element range, where they
// saturate. Matches the CUDA backend, which rounds up via cutlass ue8m0.
inline float mx_scale_round_up(float x) {
  fp8_e8m0 s(x);
  if (s.bits < 0xFE && float(s) < x) {
    s.bits += 1;
  }
  return float(s);
}

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits = 4>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}


template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename T, typename U, int values_per_thread>
inline void load_vector(const device T* x, thread U* x_thread) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = x[i];
  }
}


template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U* x_thread, U scale) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * Dequantize<8>{}(w[i]);
    }
  }

  return scale * accum;
}


template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale,
    bool partial_rows = false,
    int results_per_simdgroup = 4>
METAL_FUNC void fp_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  // Reuse the last valid row for reads beyond the output tile.
  int last_row = results_per_simdgroup - 1;
  if constexpr (partial_rows) {
    if (out_row >= out_vec_size) {
      return;
    }
    last_row = min(last_row, out_vec_size - 1 - out_row);
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      int src = row;
      if constexpr (partial_rows) {
        src = min(row, last_row);
      }
      auto wl = (const device uint8_t*)(ws + src * in_vec_size_w);
      const device auto* sl = scales + src * in_vec_size_g;

      U s = dequantize_scale<U, group_size>(sl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }

  float inv_scale_enc = 1.0f;
  if constexpr (has_global_scale) {
    inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row <= last_row) {
      if constexpr (has_global_scale) {
        y[row] = static_cast<T>(result[row] * inv_scale_enc);
      } else {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}


"""
