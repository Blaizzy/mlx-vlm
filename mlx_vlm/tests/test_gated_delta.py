"""Qwen3.5 gated delta kernels: the packed Dk=128 kernel against the generic one."""

import mlx.core as mx
import pytest

import mlx_vlm.models.qwen3_5.gated_delta as gd

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="gated delta kernels are Metal only"
)


def _normed(shape, D, dtype):
    x = mx.random.normal(shape)
    return (mx.fast.rms_norm(x, None, 1e-6) * D**-0.5).astype(dtype)


def _inputs(B, T, Hk, Hv, Dk, Dv, dtype):
    mx.random.seed(3)
    q = _normed((B, T, Hk, Dk), Dk, dtype)
    k = _normed((B, T, Hk, Dk), Dk, dtype)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    # decays in (0, 1], as produced by compute_g
    g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * 0.2).astype(mx.float32)
    beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
    state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def _generic(q, k, v, g, beta, state):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return gd._gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


@pytest.mark.parametrize(
    "B, Hk, Hv, Dv, dtype",
    [
        (1, 16, 32, 128, mx.bfloat16),  # Qwen3.5 / 3.6
        (1, 16, 48, 128, mx.bfloat16),  # 48 value heads
        (2, 16, 32, 128, mx.bfloat16),  # batched
        (1, 8, 8, 128, mx.bfloat16),  # Hv == Hk
        (1, 16, 32, 128, mx.float16),
        (1, 16, 32, 128, mx.float32),
        (1, 8, 16, 64, mx.bfloat16),  # Dv != Dk
    ],
)
@pytest.mark.parametrize("T", [2, 7, 64, 257])  # T=1 stays on the generic kernel
def test_packed_matches_generic_bitwise(B, Hk, Hv, Dv, dtype, T):
    # The packed kernel uses the same ascending butterfly that simd_sum lowers
    # to on Apple GPUs, so routing to it does not change any output bit.
    args = _inputs(B, T, Hk, Hv, 128, Dv, dtype)
    y_p, s_p = gd.gated_delta_kernel(*args, None)
    y_g, s_g = _generic(*args)
    mx.eval(y_p, s_p, y_g, s_g)
    assert mx.array_equal(y_p, y_g)
    assert mx.array_equal(s_p, s_g)


def _spy_kernels(monkeypatch):
    calls = {"packed": 0, "generic": 0}
    for name, key in (
        ("_gated_delta_kernel_packed", "packed"),
        ("_gated_delta_kernel", "generic"),
    ):
        original = getattr(gd, name)

        def spy(*a, _o=original, _k=key, **kw):
            calls[_k] += 1
            return _o(*a, **kw)

        monkeypatch.setattr(gd, name, spy)
    return calls


@pytest.mark.parametrize("B, T", [(1, 2), (1, 64), (4, 7)])
def test_multi_token_calls_select_packed_kernel(monkeypatch, B, T):
    calls = _spy_kernels(monkeypatch)
    mx.eval(*gd.gated_delta_kernel(*_inputs(B, T, 16, 32, 128, 128, mx.bfloat16), None))
    assert calls == {"packed": 1, "generic": 0}


@pytest.mark.parametrize("B", [1, 4])
def test_decode_keeps_generic_kernel(monkeypatch, B):
    calls = _spy_kernels(monkeypatch)
    mx.eval(*gd.gated_delta_kernel(*_inputs(B, 1, 16, 32, 128, 128, mx.bfloat16), None))
    assert calls == {"packed": 0, "generic": 1}


def test_packed_matches_ops_reference():
    q, k, v, g, beta, state = _inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
    y_p, s_p = gd.gated_delta_kernel(q, k, v, g, beta, state, None)
    y_r, s_r = gd.gated_delta_ops(q, k, v, g, beta, state, None)
    mx.eval(y_p, s_p, y_r, s_r)
    assert _rel_l2(y_p, y_r) < 2e-3
    assert _rel_l2(s_p, s_r) < 2e-3


def test_mask_and_vector_gate_keep_generic_kernels(monkeypatch):
    calls = {"packed": 0}
    original = gd._gated_delta_kernel_packed

    def spy(*a, **kw):
        calls["packed"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(gd, "_gated_delta_kernel_packed", spy)
    q, k, v, g, beta, state = _inputs(2, 33, 8, 16, 128, 128, mx.bfloat16)
    mask = mx.arange(33)[None] < mx.array([[29], [17]])
    y_k, s_k = gd.gated_delta_kernel(q, k, v, g, beta, state, mask)
    y_r, s_r = gd.gated_delta_ops(q, k, v, g, beta, state, mask)
    valid = mask[..., None, None]
    assert _rel_l2(mx.where(valid, y_k, 0), mx.where(valid, y_r, 0)) < 2e-3
    assert _rel_l2(s_k, s_r) < 2e-3
    g4 = mx.exp(-mx.random.uniform(shape=(2, 33, 16, 128)) * 0.2).astype(mx.float32)
    y_k, s_k = gd.gated_delta_kernel(q, k, v, g4, beta, state, None)
    y_r, s_r = gd.gated_delta_ops(q, k, v, g4, beta, state, None)
    assert _rel_l2(y_k, y_r) < 2e-3
    assert _rel_l2(s_k, s_r) < 2e-3
    assert calls["packed"] == 0
