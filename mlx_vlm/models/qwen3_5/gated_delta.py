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


def _invert_unit_lower(Tmat: mx.array, C: int) -> mx.array:
    """Inverse of a batched unit lower-triangular matrix ``Tmat`` (``[..., C, C]``,
    unit diagonal) via forward substitution, batched over all chunks/heads.

    Sequential in C (small, fixed) but numerically stable. A log-depth
    repeated-squaring (Neumann) inverse is faster but overflows when N has
    O(1) entries — exactly the regime here, since correlated keys give KK
    off-diagonals ~1."""
    X = mx.broadcast_to(mx.eye(C, dtype=mx.float32), Tmat.shape) + mx.zeros_like(Tmat)
    rows = [X[..., 0:1, :]]
    for i in range(1, C):
        Tij = Tmat[..., i : i + 1, :i]
        Xj = mx.concatenate(rows, axis=-2)
        rows.append(X[..., i : i + 1, :] - Tij @ Xj)
    return mx.concatenate(rows, axis=-2)


def gated_delta_chunked(q, k, v, g, beta, state, mask=None, C: int = 64):
    """Chunked parallel gated-delta-rule prefill (scalar per-head gating).

    Mathematically equivalent to the sequential ``gated_delta_ops`` recurrence
    but replaces the per-token loop (O(T) sequential ops) with a chunked scan:
    parallel intra-chunk matmuls + a C-length triangular solve + a T/C-step
    inter-chunk state scan. ~20x faster than the per-token loop on CUDA, where
    no fused Metal kernel is available. Validated to rel-err < 1e-3 (fp32) vs
    the reference, well within bf16 precision.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = v.dtype
    if (rf := Hv // Hk) > 1:
        q = mx.repeat(q, rf, -2)
        k = mx.repeat(k, rf, -2)
    # masked (padding) positions: no decay, no update -> g=1, beta=0
    if mask is not None:
        m = mask[..., None]
        g = mx.where(m, g, 1.0)
        beta = mx.where(m, beta, 0.0)

    pad = (C - T % C) % C
    if pad:
        q = mx.concatenate([q, mx.zeros((B, pad, Hv, Dk), q.dtype)], axis=1)
        k = mx.concatenate([k, mx.zeros((B, pad, Hv, Dk), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((B, pad, Hv, Dv), v.dtype)], axis=1)
        g = mx.concatenate([g, mx.ones((B, pad, Hv), g.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((B, pad, Hv), beta.dtype)], axis=1)
    Tp = T + pad
    nC = Tp // C

    def rc(x, D):
        return x.reshape(B, nC, C, Hv, D).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    q, k, v = rc(q, Dk), rc(k, Dk), rc(v, Dv)  # [B,Hv,nC,C,D]
    g = g.reshape(B, nC, C, Hv).transpose(0, 3, 1, 2)  # [B,Hv,nC,C]
    beta = beta.reshape(B, nC, C, Hv).transpose(0, 3, 1, 2)

    # clip g off 0: compute_g can underflow to exactly 0.0 in fp32, and
    # log(0)=-inf would poison the decay ratios with NaN.
    lcg = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)  # log cumulative decay
    cumg = mx.exp(lcg)
    lower_incl = mx.tril(mx.ones((C, C), mx.float32), 0)
    strict_lower = mx.tril(mx.ones((C, C), mx.float32), -1)
    # decay_ratio[i,j]=cumg_i/cumg_j; mask exponent to lower triangle BEFORE exp
    # (upper triangle would overflow since g<1, and inf*0=nan).
    diff = mx.where(lower_incl > 0, lcg[..., :, None] - lcg[..., None, :], -1e30)
    dr = mx.exp(diff)

    KK = k @ mx.swapaxes(k, -1, -2)
    A = beta[..., :, None] * dr * KK * strict_lower
    Tinv = _invert_unit_lower(mx.eye(C, dtype=mx.float32) + A, C)

    kbeta = (beta * cumg)[..., :, None] * k
    U0 = Tinv @ (beta[..., :, None] * v)  # [B,Hv,nC,C,Dv]
    Kt = Tinv @ kbeta  # [B,Hv,nC,C,Dk]
    M = dr * (q @ mx.swapaxes(k, -1, -2)) * lower_incl
    Qeff = cumg[..., :, None] * q - M @ Kt
    MU0 = M @ U0
    cumg_last = cumg[..., -1]
    ratio_last = mx.exp(lcg[..., -1, None] - lcg)  # cumg_last/cumg_i

    S = (
        state.astype(mx.float32)
        if state is not None
        else mx.zeros((B, Hv, Dv, Dk), mx.float32)
    )
    ys = []
    for c in range(nC):
        St = mx.swapaxes(S, -1, -2)  # [B,Hv,Dk,Dv]
        ys.append(MU0[:, :, c] + Qeff[:, :, c] @ St)  # [B,Hv,C,Dv]
        Uc = U0[:, :, c] - Kt[:, :, c] @ St
        Uscaled = ratio_last[:, :, c][..., None] * Uc
        S = (
            cumg_last[:, :, c][..., None, None] * S
            + mx.swapaxes(Uscaled, -1, -2) @ k[:, :, c]
        )
    Y = mx.stack(ys, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, Tp, Hv, Dv)[:, :T]
    Y = Y.astype(in_dtype)
    # Schedule this (large, dynamically-shaped) prefill subgraph for evaluation
    # so it is NOT folded into the server's captured CUDA graph — that capture
    # chokes on it (cudaGraphAddDependencies). async_eval breaks the graph at
    # this boundary without blocking the Python thread, letting graph-building
    # for the next layer overlap with GPU compute. Decode (T==1) never calls
    # this path, so its CUDA graphs stay intact.
    try:
        mx.async_eval(Y, S)
    except ValueError as exc:
        if "Not allowed inside a graph transformation" not in str(exc):
            raise
    return Y, S


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
        # Prefill (T>1) with scalar gating: use the chunked parallel scan, which
        # is ~20x faster than the per-token loop on backends without a fused
        # kernel (e.g. CUDA). Decode (T==1) and vectorized gating fall back to
        # the reference ops.
        if q.shape[1] > 1 and g.ndim == 3:
            return gated_delta_chunked(q, k, v, g, beta, state, mask)
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
