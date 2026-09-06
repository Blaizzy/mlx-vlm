from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..exact_speculative_verify import (
    exact_speculative_verify_dense_available,
    exact_speculative_verify_weight,
)


def _quantized_head_header(bits: int, group_size: int) -> str:
    return r"""
    using namespace metal;

    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 8;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
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
      if (BITS == 4) {
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
""".replace("__BITS__", str(bits)).replace("__GS__", str(group_size))


_QUANTIZED_HEAD_QMV_SOURCE = r"""
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


@lru_cache(maxsize=None)
def _quantized_head_qmv_kernel(
    bits,
    group_size,
    dtype,
    verify_t,
    k_size,
    n_size,
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "glm_moe_dsa_target_verify_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_quantized_head_header(bits, group_size),
        source=_QUANTIZED_HEAD_QMV_SOURCE,
    )


def _can_verify_quantized_head(linear) -> bool:
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or linear.bits not in (4, 5, 8)
        or linear.mode != "affine"
        or linear.biases is None
        or linear.scales.dtype not in (mx.bfloat16, mx.float16)
        or linear.biases.dtype != linear.scales.dtype
    ):
        return False
    input_size = linear.weight.shape[1] * 32 // linear.bits
    output_size = linear.weight.shape[0]
    return input_size % 512 == 0 and output_size % 8 == 0


def _quantized_head_logits(linear, x: mx.array) -> Optional[mx.array]:
    if (
        not _can_verify_quantized_head(linear)
        or x.ndim != 3
        or x.shape[1] < 1
        or x.dtype != linear.scales.dtype
    ):
        return None

    batch, length, input_size = x.shape
    if input_size != linear.weight.shape[1] * 32 // linear.bits:
        return None
    output_size = linear.weight.shape[0]
    kernel = _quantized_head_qmv_kernel(
        linear.bits,
        linear.group_size,
        x.dtype,
        length,
        input_size,
        output_size,
    )
    logits = kernel(
        inputs=[mx.contiguous(x), linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(length)),
            ("K_SIZE", int(input_size)),
            ("N_SIZE", int(output_size)),
        ],
        grid=(32, output_size // 8, batch),
        threadgroup=(32, 2, 1),
        output_shapes=[(batch, length, output_size)],
        output_dtypes=[x.dtype],
    )[0]
    return logits + linear["bias"] if "bias" in linear else logits


def _timewise_linear(linear, x: mx.array) -> mx.array:
    return mx.concatenate(
        [linear(x[:, index : index + 1]) for index in range(x.shape[1])], axis=1
    )


def verify_logits(language_model, normed_hidden: mx.array) -> mx.array:
    head = language_model.lm_head
    if isinstance(head, nn.QuantizedLinear):
        logits = _quantized_head_logits(head, normed_hidden)
        if logits is not None:
            return logits
        return _timewise_linear(head, normed_hidden)
    if (
        exact_speculative_verify_dense_available()
        and isinstance(head, nn.Linear)
        and "bias" not in head
    ):
        logits = exact_speculative_verify_weight(head.weight, normed_hidden)
        if logits is not None:
            return logits
    return _timewise_linear(head, normed_hidden)


class GlmMoeDsaExactSpeculativeVerifier:
    """Verify an MTP proposal using the target's pre-final-norm hidden states."""

    @staticmethod
    def _capture_hidden(
        language_model,
        inputs: mx.array,
        cache: Any,
        inputs_embeds: Optional[mx.array],
        capture_layer_ids: list[int],
        hidden_sink: list[mx.array],
    ) -> mx.array:
        model = language_model.model
        if model.pipeline_size != 1:
            raise ValueError(
                "glm_moe_dsa layer capture does not support pipeline parallelism."
            )
        if len(set(capture_layer_ids)) != len(capture_layer_ids) or any(
            not isinstance(layer_id, int)
            or layer_id < 0
            or layer_id >= model.num_layers
            for layer_id in capture_layer_ids
        ):
            raise ValueError(
                "capture_layer_ids must be unique layer indices inside the model."
            )

        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * model.num_layers
        mask = create_attention_mask(
            hidden, cache[0][0] if cache[0] else None, return_array=True
        )
        capture_set = set(capture_layer_ids)
        prev_topk_indices = None
        for index in range(model.num_layers):
            layer_index = model.start_idx + index
            hidden, prev_topk_indices = model.layers[layer_index](
                hidden, mask, cache[index], prev_topk_indices
            )
            if layer_index in capture_set:
                hidden_sink.append(hidden)
        return model.norm(hidden)

    def __call__(
        self,
        language_model,
        inputs: mx.array,
        *,
        cache: Any = None,
        inputs_embeds: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        hidden_sink = [] if capture_layer_ids is not None else None
        if capture_layer_ids:
            hidden = self._capture_hidden(
                language_model,
                inputs,
                cache,
                inputs_embeds,
                capture_layer_ids,
                hidden_sink,
            )
        else:
            hidden = language_model.model(
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
            )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(hidden)
        logits = None if skip_logits else verify_logits(language_model, hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )


__all__ = ["GlmMoeDsaExactSpeculativeVerifier", "verify_logits"]
