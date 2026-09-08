from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..exact_speculative_verify import exact_speculative_verify_weight
from ..qwen3_5.speculative_verifier import Qwen3_5ExactSpeculativeVerifier
from ..ssm import compute_dt, ssm_update

_MAMBA_VERIFY_SOURCE = r"""
    uint lane = thread_position_in_grid.x;
    uint d_idx = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint batch_idx = n / H;
    uint head_idx = n % H;
    uint group_idx = head_idx / HEADS_PER_GROUP;
    uint state_idx = ((batch_idx * H + head_idx) * DH + d_idx) * DS;
    constexpr int N_PER_THREAD = DS / 32;

    U local_state[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        local_state[i] = state_in[state_idx + s_idx];
    }

    float A = -fast::exp(static_cast<float>(A_log[head_idx]));
    for (int token = 0; token < VERIFY_T; ++token) {
        uint x_idx = ((batch_idx * VERIFY_T + token) * H + head_idx) * DH + d_idx;
        uint bc_idx = ((batch_idx * VERIFY_T + token) * GROUPS + group_idx) * DS;
        uint dt_idx = (batch_idx * VERIFY_T + token) * H + head_idx;
        float dt_value = static_cast<float>(dt[dt_idx]);
        float decay = fast::exp(A * dt_value);
        float x_value = static_cast<float>(X[x_idx]);
        float acc = 0.0f;

        for (int i = 0; i < N_PER_THREAD; ++i) {
            uint s_idx = N_PER_THREAD * lane + i;
            float next_state =
                decay * static_cast<float>(local_state[i]) +
                x_value * dt_value * static_cast<float>(B[bc_idx + s_idx]);
            local_state[i] = static_cast<U>(next_state);
            acc += next_state * static_cast<float>(C[bc_idx + s_idx]);
        }

        acc = simd_sum(acc);
        if (lane == 0) {
            out[x_idx] = static_cast<T>(
                acc + x_value * static_cast<float>(D[head_idx])
            );
        }
    }

    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        state_out[state_idx + s_idx] = local_state[i];
    }
"""

_MAMBA_STATE_SOURCE = r"""
    uint lane = thread_position_in_grid.x;
    uint d_idx = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint batch_idx = n / H;
    uint head_idx = n % H;
    uint group_idx = head_idx / HEADS_PER_GROUP;
    uint state_idx = ((batch_idx * H + head_idx) * DH + d_idx) * DS;
    constexpr int N_PER_THREAD = DS / 32;

    U local_state[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        local_state[i] = state_in[state_idx + s_idx];
    }

    float A = -fast::exp(static_cast<float>(A_log[head_idx]));
    for (int token = 0; token < VERIFY_T; ++token) {
        uint x_idx = ((batch_idx * VERIFY_T + token) * H + head_idx) * DH + d_idx;
        uint b_idx = ((batch_idx * VERIFY_T + token) * GROUPS + group_idx) * DS;
        uint dt_idx = (batch_idx * VERIFY_T + token) * H + head_idx;
        float dt_value = static_cast<float>(dt[dt_idx]);
        float decay = fast::exp(A * dt_value);
        float x_value = static_cast<float>(X[x_idx]);

        for (int i = 0; i < N_PER_THREAD; ++i) {
            uint s_idx = N_PER_THREAD * lane + i;
            float next_state =
                decay * static_cast<float>(local_state[i]) +
                x_value * dt_value * static_cast<float>(B[b_idx + s_idx]);
            local_state[i] = static_cast<U>(next_state);
        }
    }

    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        state_out[state_idx + s_idx] = local_state[i];
    }
"""

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
            "nemotron_h_nvfp4_argmax_"
            f"t{verify_length}_k{input_size}_n{output_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales"],
        output_names=["tile_values", "tile_indices"],
        header=_NVFP4_ARGMAX_HEADER,
        source=_NVFP4_ARGMAX_SOURCE,
    )


def _nvfp4_argmax(linear, hidden: mx.array) -> Optional[mx.array]:
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


@lru_cache(maxsize=None)
def _mamba_verify_kernel(
    input_dtype,
    state_dtype,
    verify_length,
    num_heads,
    head_dim,
    state_size,
    num_groups,
):
    input_name = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}.get(
        input_dtype, "unk"
    )
    state_name = {
        mx.bfloat16: "bf16",
        mx.float16: "fp16",
        mx.float32: "fp32",
    }.get(state_dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "nemotron_h_target_verify_mamba_"
            f"t{verify_length}_h{num_heads}_dh{head_dim}_ds{state_size}_"
            f"g{num_groups}_{input_name}_{state_name}"
        ),
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=_MAMBA_VERIFY_SOURCE,
    )


@lru_cache(maxsize=None)
def _mamba_state_kernel(
    input_dtype,
    state_dtype,
    verify_length,
    num_heads,
    head_dim,
    state_size,
    num_groups,
):
    input_name = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}.get(
        input_dtype, "unk"
    )
    state_name = {
        mx.bfloat16: "bf16",
        mx.float16: "fp16",
        mx.float32: "fp32",
    }.get(state_dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "nemotron_h_target_replay_mamba_"
            f"t{verify_length}_h{num_heads}_dh{head_dim}_ds{state_size}_"
            f"g{num_groups}_{input_name}_{state_name}"
        ),
        input_names=["X", "A_log", "B", "dt", "state_in"],
        output_names=["state_out"],
        source=_MAMBA_STATE_SOURCE,
    )


def replay_mamba_state(rollback_state: dict[str, mx.array], length: int) -> mx.array:
    hidden_states = rollback_state["hidden_states"][:, :length]
    B = rollback_state["B"][:, :length]
    dt = rollback_state["dt"][:, :length]
    state = rollback_state["state"]
    A_log = rollback_state["A_log"]
    batch, _, num_heads, head_dim = hidden_states.shape
    num_groups, state_size = B.shape[-2:]
    kernel = _mamba_state_kernel(
        hidden_states.dtype,
        state.dtype,
        length,
        num_heads,
        head_dim,
        state_size,
        num_groups,
    )
    return kernel(
        inputs=[hidden_states, A_log, B, dt, state],
        template=[
            ("T", hidden_states.dtype),
            ("U", state.dtype),
            ("VERIFY_T", int(length)),
            ("H", int(num_heads)),
            ("DH", int(head_dim)),
            ("DS", int(state_size)),
            ("GROUPS", int(num_groups)),
            ("HEADS_PER_GROUP", int(num_heads // num_groups)),
        ],
        grid=(32, head_dim, batch * num_heads),
        threadgroup=(32, min(8, head_dim), 1),
        output_shapes=[state.shape],
        output_dtypes=[state.dtype],
    )[0]


def _mamba_update_timewise(
    hidden_states,
    A_log,
    B,
    C,
    D,
    dt,
    dt_bias,
    state,
    time_step_limit,
    mask,
):
    outputs = []
    states = []
    for position in range(hidden_states.shape[1]):
        output, state = ssm_update(
            hidden_states[:, position : position + 1],
            A_log,
            B[:, position : position + 1],
            C[:, position : position + 1],
            D,
            dt[:, position : position + 1],
            dt_bias,
            state,
            time_step_limit,
            None if mask is None else mask[:, position : position + 1],
        )
        outputs.append(output)
        states.append(state[:, None])
    return mx.concatenate(outputs, axis=1), state, mx.concatenate(states, axis=1)


def _mamba_update_exact(
    hidden_states,
    A_log,
    B,
    C,
    D,
    dt,
    dt_bias,
    state,
    time_step_limit,
    mask,
):
    batch, length, num_heads, head_dim = hidden_states.shape
    num_groups, state_size = B.shape[-2:]
    if (
        state is None
        or mask is not None
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or state_size % 32
        or hidden_states.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or state.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return _mamba_update_timewise(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            time_step_limit,
            mask,
        )

    dt = compute_dt(dt, dt_bias, time_step_limit)
    kernel = _mamba_verify_kernel(
        hidden_states.dtype,
        state.dtype,
        length,
        num_heads,
        head_dim,
        state_size,
        num_groups,
    )
    output, final_state = kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("T", hidden_states.dtype),
            ("U", state.dtype),
            ("VERIFY_T", int(length)),
            ("H", int(num_heads)),
            ("DH", int(head_dim)),
            ("DS", int(state_size)),
            ("GROUPS", int(num_groups)),
            ("HEADS_PER_GROUP", int(num_heads // num_groups)),
        ],
        grid=(32, head_dim, batch * num_heads),
        threadgroup=(32, min(8, head_dim), 1),
        output_shapes=[
            hidden_states.shape,
            state.shape,
        ],
        output_dtypes=[hidden_states.dtype, state.dtype],
    )
    rollback_state = {
        "hidden_states": hidden_states,
        "A_log": A_log,
        "B": B,
        "dt": dt,
        "state": state,
    }
    return output, final_state, rollback_state


class NemotronHExactSpeculativeVerifier(Qwen3_5ExactSpeculativeVerifier):
    """Run Nemotron-H verification with singleton-equivalent Metal kernels."""

    def quantized_argmax(
        self,
        linear,
        hidden: mx.array,
        token_mask: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        output = super().quantized_argmax(linear, hidden, token_mask=token_mask)
        if output is not None or token_mask is not None:
            return output
        return _nvfp4_argmax(linear, hidden)

    def _linear(self, linear, hidden: mx.array) -> mx.array:
        if hidden.ndim == 3 and hidden.shape[1] > 1 and hidden.dtype == mx.float32:
            return mx.concatenate(
                [
                    linear(hidden[:, position : position + 1])
                    for position in range(hidden.shape[1])
                ],
                axis=1,
            )
        return super()._linear(linear, hidden)

    def _linears(self, linears, hidden: mx.array):
        if hidden.ndim == 3 and hidden.shape[1] > 1 and hidden.dtype == mx.float32:
            return tuple(self._linear(linear, hidden) for linear in linears)
        return super()._linears(linears, hidden)

    def linear(self, linear, hidden: mx.array) -> mx.array:
        return self._linear(linear, hidden)

    def _attention(self, attention, hidden, mask, cache):
        batch, length, _ = hidden.shape
        queries, keys, values = self._linears(
            (attention.q_proj, attention.k_proj, attention.v_proj), hidden
        )
        queries = queries.reshape(
            batch, length, attention.num_heads, attention.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch, length, attention.num_key_value_heads, attention.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, attention.num_key_value_heads, attention.head_dim
        ).transpose(0, 2, 1, 3)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if length > 1:
            prefix_length = keys.shape[-2] - length
            output = mx.concatenate(
                [
                    scaled_dot_product_attention(
                        queries[:, :, position : position + 1],
                        keys[:, :, : prefix_length + position + 1],
                        values[:, :, : prefix_length + position + 1],
                        cache=cache,
                        scale=attention.scale,
                        mask=(
                            mask[
                                ...,
                                position : position + 1,
                                : prefix_length + position + 1,
                            ]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                    )
                    for position in range(length)
                ],
                axis=2,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=attention.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output)

    def _mamba(self, mixer, hidden, mask, cache, state_sink):
        projected = self._linear(mixer.in_proj, hidden)
        gate, conv_input, dt = mixer._split_projected_states(projected)
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (hidden.shape[0], mixer.conv_kernel_size - 1, mixer.conv_dim),
                dtype=hidden.dtype,
            )
        padded_input = mx.concatenate([conv_state, conv_input], axis=1)
        if cache is not None:
            keep = mixer.conv_kernel_size - 1
            if cache.lengths is not None:
                total = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, total - keep)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(padded_input[:, -keep:])

        conv_output = nn.silu(mixer.conv1d(padded_input))
        hidden_ssm, B, C = mx.split(
            conv_output,
            [
                mixer.intermediate_size,
                mixer.intermediate_size + mixer.n_groups * mixer.ssm_state_size,
            ],
            axis=-1,
        )
        batch, length, _ = hidden_ssm.shape
        hidden_ssm = hidden_ssm.reshape(batch, length, mixer.num_heads, mixer.head_dim)
        B = B.reshape(batch, length, mixer.n_groups, mixer.ssm_state_size)
        C = C.reshape(batch, length, mixer.n_groups, mixer.ssm_state_size)
        state = cache[1] if cache is not None else None
        output, state, rollback_state = _mamba_update_exact(
            hidden_ssm,
            mixer.A_log,
            B,
            C,
            mixer.D.astype(hidden_ssm.dtype),
            dt,
            mixer.dt_bias,
            state,
            mixer.time_step_limit,
            mask,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(length)
        state_sink.append((rollback_state, padded_input, mixer.conv_kernel_size))
        output = output.reshape(batch, length, mixer.intermediate_size)
        output = mixer.norm(output, gate)
        return self._linear(mixer.out_proj, output)

    def _mlp(self, mlp, hidden):
        return self._linear(mlp.down_proj, nn.relu2(self._linear(mlp.up_proj, hidden)))

    def _switch_mlp(self, switch_mlp, hidden, indices):
        if hidden.ndim != 3 or hidden.shape[1] <= 1:
            return switch_mlp(hidden, indices)
        batch, length, width = hidden.shape
        top_k = indices.shape[-1]
        hidden = hidden.reshape(batch * length, width)
        indices = indices.reshape(batch * length, top_k)
        hidden = mx.expand_dims(hidden, (-2, -3))
        hidden = switch_mlp.fc1(hidden, indices, sorted_indices=False)
        hidden = switch_mlp.activation(hidden)
        hidden = switch_mlp.fc2(hidden, indices, sorted_indices=False)
        return hidden.squeeze(-2).reshape(batch, length, top_k, -1)

    def _moe(self, moe, hidden):
        from .language import group_expert_select

        gate_logits = (
            None
            if hidden.dtype == mx.float32
            else exact_speculative_verify_weight(moe.gate.weight, hidden)
        )
        if gate_logits is None:
            gate_logits = mx.concatenate(
                [
                    hidden[:, position : position + 1] @ moe.gate.weight.T
                    for position in range(hidden.shape[1])
                ],
                axis=1,
            )
        indices, scores = group_expert_select(
            gate_logits,
            moe.gate.e_score_correction_bias,
            moe.gate.top_k,
            moe.gate.n_group,
            moe.gate.topk_group,
            moe.gate.routed_scaling_factor,
            moe.gate.norm_topk_prob,
        )
        residual = hidden
        if moe.moe_latent_size is not None:
            hidden = self._linear(moe.fc1_latent_proj, hidden)
        output = self._switch_mlp(moe.switch_mlp, hidden, indices)
        output = (output * scores[..., None]).sum(axis=-2).astype(output.dtype)
        if moe.moe_latent_size is not None:
            output = self._linear(moe.fc2_latent_proj, output)
        if moe.config.n_shared_experts is not None:
            output = output + self._mlp(moe.shared_experts, residual)
        return output

    def _layer(self, layer, hidden, mask, cache, state_sink):
        normed = layer.norm(hidden)
        if layer.block_type == "M":
            output = self._mamba(layer.mixer, normed, mask, cache, state_sink)
        elif layer.block_type == "*":
            output = self._attention(layer.mixer, normed, mask, cache)
        elif layer.block_type == "-":
            output = self._mlp(layer.mixer, normed)
        else:
            output = self._moe(layer.mixer, normed)
        return hidden + output

    def _model(
        self,
        model,
        inputs,
        cache,
        inputs_embeds,
        capture_layer_ids,
        hidden_sink,
        state_sink,
    ):
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif model.with_embeddings:
            hidden = model.embeddings(inputs)
        else:
            raise ValueError("This Nemotron-H backbone has no token embedding table")

        stateful_layers = sum(layer.block_type in {"M", "*"} for layer in model.layers)
        if cache is None:
            cache = [None] * stateful_layers
        has_attention = any(layer.block_type == "*" for layer in model.layers)
        has_mamba = any(layer.block_type == "M" for layer in model.layers)
        attention_cache = cache[model.fa_idx] if has_attention else None
        mamba_cache = cache[model.ssm_idx] if has_mamba else None
        attention_mask = create_attention_mask(hidden, attention_cache)
        mamba_mask = create_ssm_mask(hidden, mamba_cache)
        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        cache_index = 0
        for index, layer in enumerate(model.layers):
            layer_cache = cache[cache_index] if layer.block_type in {"M", "*"} else None
            if layer.block_type in {"M", "*"}:
                cache_index += 1
            mask = attention_mask if layer.block_type == "*" else mamba_mask
            hidden = self._layer(layer, hidden, mask, layer_cache, state_sink)
            if hidden_sink is not None and index in capture_set:
                hidden_sink.append(hidden)
        return model.norm_f(hidden)

    def __call__(
        self,
        language_model,
        inputs: Optional[mx.array],
        *,
        cache: Optional[list[Any]] = None,
        inputs_embeds: Optional[mx.array] = None,
        capture_layer_ids: Optional[list[int]] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        if inputs is None and inputs_embeds is None:
            raise ValueError("Provide either inputs or inputs_embeds")
        hidden_sink = [] if capture_layer_ids is not None else None
        state_sink = []
        hidden = self._model(
            language_model.backbone,
            inputs,
            cache,
            inputs_embeds,
            capture_layer_ids,
            hidden_sink,
            state_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(hidden)
        logits = None if skip_logits else self._linear(language_model.lm_head, hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=state_sink,
            shared_kv_states={} if return_shared_kv else None,
        )


__all__ = ["NemotronHExactSpeculativeVerifier", "replay_mamba_state"]
