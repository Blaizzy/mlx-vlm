"""Opt-in batch-invariant projections for batched generation.

``mx.quantized_matmul`` flattens a row-contiguous ``(B, L, K)`` input to
``M = B * L`` rows and picks its Metal kernel from ``M``: a single decode row
runs ``qmv``, 2 <= M < the qmv batch limit runs ``qmv_wide`` (Apple M3 and
later, affine quantization), and larger M runs ``qmm``. The kernels reduce in
different orders, so a row's bf16 output depends on how many other rows share
the call, and greedy batched decoding can drift from batch-size-1 decoding.

When enabled, :func:`linear` makes every row of a batched projection equal to
running that row alone: batched decode (``L == 1``) on a supported affine
``QuantizedLinear`` goes through the exact verify kernel, which matches the
single-row ``qmv`` result bit for bit; any other batched shape (prefill,
unsupported layers) runs row by row.

The switch is a :class:`contextvars.ContextVar`, set by ``BatchGenerator``
only for the duration of its own ``next()`` call, so it never leaks between
generators, threads, or past ``close()``. When it is off, :func:`linear` is a
flag check followed by the plain ``layer(x)`` call.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from .quantized_verifier import optimized_affine_linear as _optimized_affine_linear

_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mlx_vlm_batch_invariant", default=False
)


def is_enabled() -> bool:
    """Return whether batch-invariant projections are active in this context."""
    return _ENABLED.get()


@contextlib.contextmanager
def batch_invariant(enabled: bool = True):
    """Enable (or explicitly disable) batch-invariant projections in a block."""
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


def linear(layer: Callable[[mx.array], mx.array], x: mx.array) -> mx.array:
    """Apply ``layer`` to ``x``; when enabled, each batch row matches B == 1.

    ``layer`` is any callable projection (``nn.Linear``, ``nn.QuantizedLinear``,
    or e.g. ``embedding.as_linear``). With the switch off, or for B == 1, this
    is exactly ``layer(x)``.
    """
    if not _ENABLED.get() or x.ndim != 3 or x.shape[0] == 1:
        return layer(x)
    B, L, K = x.shape
    if L == 1 and isinstance(layer, nn.QuantizedLinear):
        out = _optimized_affine_linear(layer, x.reshape(1, B, K))
        if out is not None:
            return out.reshape(B, 1, -1)
    return mx.concatenate([layer(x[i : i + 1]) for i in range(B)], axis=0)
