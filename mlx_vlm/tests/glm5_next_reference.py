"""Independent KDA recurrence for parity tests."""

from typing import Optional

import mlx.core as mx


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Reference L2 normalization used by the GLM KDA parity oracle."""
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    """Readable recurrent KDA reference used for implementation parity tests."""
    dtype = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query = query * (key_dim**-0.5)
    if state is None:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outputs = []
    for index in range(length):
        q_i = query[:, index]
        k_i = key[:, index]
        v_i = value[:, index]
        g_i = mx.exp(g[:, index])[..., None]
        beta_i = beta[:, index][..., None]
        state = state * g_i
        memory = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - memory) * beta_i
        state = state + k_i[..., None] * delta[..., None, :]
        outputs.append((state * q_i[..., None]).sum(axis=-2))
    return mx.stack(outputs, axis=1).astype(dtype), state
