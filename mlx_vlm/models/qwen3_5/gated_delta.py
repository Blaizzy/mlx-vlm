from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def _compute_g_beta(A_log, a, b, dt_bias):
    return compute_g(A_log, a, dt_bias), mx.sigmoid(b)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
):
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    if state is None:
        B, _, _Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)


def _make_gated_delta_with_states_kernel(has_mask: bool = False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        states += ((b_idx * T * Hv + hv_idx) * Dv) * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto states_ = states + dv_idx * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
            if ({mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                    y[dv_idx] = static_cast<InT>(out);
                }}
            }} else {{
                y[dv_idx] = static_cast<InT>(0);
            }}

            if (t < StateT) {{
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    states_[s_idx] = static_cast<StT>(state[i]);
                }}
            }}

            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            if (t < StateT) {{
                states_ += Hv * Dv * Dk;
            }}
            g_ += Hv;
            beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"qwen3_5_gated_delta_with_states{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "states"],
        source=source,
    )


_gated_delta_with_states_kernel = _make_gated_delta_with_states_kernel(False)
_gated_delta_with_states_kernel_masked = _make_gated_delta_with_states_kernel(True)


def _gated_delta_with_states_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    B, T, Hk, _Dk = q.shape
    Hv = v.shape[-2]
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    states = []
    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        y = (state * q[:, t, :, None, :]).sum(axis=-1)

        if mask is not None:
            valid = mask[:, t]
            state = mx.where(valid[:, None, None, None], state, old_state)
            y = mx.where(valid[:, None, None], y, 0)

        ys.append(y.astype(q.dtype))
        states.append(state)
    stacked_states = mx.stack(states, axis=1)
    return mx.stack(ys, axis=1), state, stacked_states


def gated_delta_update_with_states(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
):
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    if state is None:
        B, _, _Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        g.ndim != 3
        or not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return _gated_delta_with_states_ops(q, k, v, g, beta, state, mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    state_steps = T

    input_type = q.dtype
    state_type = state.dtype
    kernel = _gated_delta_with_states_kernel
    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        kernel = _gated_delta_with_states_kernel_masked
        inputs.append(mask)
    if kernel is None:
        return _gated_delta_with_states_ops(q, k, v, g, beta, state, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("StateT", state_steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, state_steps, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type, state_type],
    )


def _make_gated_delta_state_kernel(has_mask: bool = False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        int valid_steps = steps[b_idx];

        for (int t = 0; t < T; ++t) {{
            if (t < valid_steps && {mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                }}
            }}

            k_ += Hk * Dk;
            v_ += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["k", "v", "g", "beta", "state_in", "steps", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"qwen3_5_gated_delta_state{suffix}",
        input_names=inputs,
        output_names=["state_out"],
        source=source,
    )


_gated_delta_state_kernel = _make_gated_delta_state_kernel(False)
_gated_delta_state_kernel_masked = _make_gated_delta_state_kernel(True)


def _make_gated_delta_accept_states_kernel():
    if not mx.metal.is_available():
        return None

    return mx.fast.metal_kernel(
        name="qwen3_5_gated_delta_accept_states",
        input_names=[
            "intermediate_states",
            "conv_input",
            "live_state",
            "live_conv",
            "accepted",
        ],
        output_names=["state_out", "conv_out"],
        source=r"""
            uint idx = thread_position_in_grid.x;

            if (idx < StateTotal) {
                uint dk = idx % Dk;
                uint t0 = idx / Dk;
                uint dv = t0 % Dv;
                t0 /= Dv;
                uint hv = t0 % Hv;
                uint row = t0 / Hv;

                int step = int(accepted[row]);
                bool use_intermediate = step >= 0 && step < T;
                StT value;
                if (use_intermediate) {
                    value = intermediate_states[
                        ((((row * T + uint(step)) * Hv + hv) * Dv + dv) * Dk + dk)
                    ];
                } else {
                    value = live_state[((row * Hv + hv) * Dv + dv) * Dk + dk];
                }
                state_out[idx] = static_cast<StT>(value);
            }

            if (idx < ConvTotal) {
                uint c = idx % C;
                uint t0 = idx / C;
                uint win = t0 % ConvW;
                uint row = t0 / ConvW;

                int step = int(accepted[row]);
                bool use_intermediate = step >= 0 && step < T;
                ConvT value;
                if (use_intermediate) {
                    value = conv_input[
                        (row * ConvInputT + uint(step) + 1 + win) * C + c
                    ];
                } else {
                    value = live_conv[(row * ConvW + win) * C + c];
                }
                conv_out[idx] = static_cast<ConvT>(value);
            }
        """,
    )


_gated_delta_accept_states_kernel = _make_gated_delta_accept_states_kernel()


def _gated_delta_state_ops(
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    steps: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    _B, T, Hk, _Dk = k.shape
    Hv = v.shape[-2]
    if (repeat_factor := Hv // Hk) > 1:
        k = mx.repeat(k, repeat_factor, -2)

    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]

        valid = steps > t
        if mask is not None:
            valid = valid & mask[:, t]
        state = mx.where(valid[:, None, None, None], state, old_state)
    return state


def _gated_delta_accept_states_ops(
    intermediate_states: mx.array,
    conv_input: mx.array,
    live_state: mx.array,
    live_conv: mx.array,
    accepted: mx.array,
    kernel_size: int,
):
    steps = [int(step) for step in accepted.tolist()]
    state_rows = []
    conv_rows = []
    state_steps = intermediate_states.shape[1]
    for row, step in enumerate(steps):
        if 0 <= step < state_steps:
            state_rows.append(intermediate_states[row, step])
            conv_rows.append(conv_input[row : row + 1, step + 1 : step + kernel_size])
        else:
            state_rows.append(live_state[row])
            conv_rows.append(live_conv[row : row + 1])
    return mx.stack(state_rows, axis=0), mx.concatenate(conv_rows, axis=0)


def gated_delta_accept_states(
    intermediate_states: mx.array,
    conv_input: mx.array,
    live_state: mx.array,
    live_conv: mx.array,
    accepted: mx.array,
    kernel_size: int,
    use_kernel: bool = True,
):
    if accepted.dtype != mx.int32:
        accepted = accepted.astype(mx.int32)

    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or _gated_delta_accept_states_kernel is None
    ):
        return _gated_delta_accept_states_ops(
            intermediate_states,
            conv_input,
            live_state,
            live_conv,
            accepted,
            kernel_size,
        )

    rows, state_steps, Hv, Dv, Dk = intermediate_states.shape
    conv_input_t = conv_input.shape[1]
    conv_dim = conv_input.shape[-1]
    conv_window = int(kernel_size) - 1
    state_total = rows * Hv * Dv * Dk
    conv_total = rows * conv_window * conv_dim
    total = max(state_total, conv_total)

    return _gated_delta_accept_states_kernel(
        inputs=[intermediate_states, conv_input, live_state, live_conv, accepted],
        template=[
            ("StT", intermediate_states.dtype),
            ("ConvT", conv_input.dtype),
            ("T", state_steps),
            ("Hv", Hv),
            ("Dv", Dv),
            ("Dk", Dk),
            ("C", conv_dim),
            ("ConvW", conv_window),
            ("ConvInputT", conv_input_t),
            ("StateTotal", state_total),
            ("ConvTotal", conv_total),
        ],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[live_state.shape, live_conv.shape],
        output_dtypes=[intermediate_states.dtype, conv_input.dtype],
    )


def gated_delta_state_update(
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array],
    steps: mx.array,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> mx.array:
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    if state is None:
        B, _, _Hk, Dk = k.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        g.ndim != 3
        or not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return _gated_delta_state_ops(k, v, g, beta, state, steps, mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    state_type = state.dtype
    kernel = _gated_delta_state_kernel
    inputs = [k, v, g, beta, state, steps, T]
    if mask is not None:
        kernel = _gated_delta_state_kernel_masked
        inputs.append(mask)
    if kernel is None:
        return _gated_delta_state_ops(k, v, g, beta, state, steps, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[state.shape],
        output_dtypes=[state_type],
    )[0]
