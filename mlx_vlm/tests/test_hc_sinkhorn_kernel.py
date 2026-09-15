"""Tests for the fused sinkhorn-split kernel used by single-pass mHC.

The kernel replaces a chain of small ops, so the thing to pin is that it agrees
with that chain rather than that it merely returns something shaped right. Each
comparison is against `_hc_split_sinkhorn_ops` on the same inputs.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.deepseek_v4.hyper_connection import (
    _hc_sinkhorn_split_kernel,
    _hc_split_sinkhorn_ops,
    hc_split_sinkhorn,
)

HC = 4
ITERS = 3
EPS = 1e-6
MIX = (2 + HC) * HC

requires_kernel = pytest.mark.skipif(
    _hc_sinkhorn_split_kernel is None, reason="Metal kernel unavailable"
)


def _inputs(batch, length, spread=1.0, seed=0):
    mx.random.seed(seed)
    mixes = mx.random.normal((batch, length, MIX)) * spread
    scale = mx.random.uniform(0.5, 2.0, (3,))
    base = mx.random.normal((MIX,))
    mx.eval(mixes, scale, base)
    return mixes, scale, base


@requires_kernel
@pytest.mark.parametrize("batch,length", [(1, 1), (1, 4), (2, 3), (1, 16)])
@pytest.mark.parametrize("spread", [1.0, 10.0])
def test_kernel_matches_the_op_path(batch, length, spread):
    mixes, scale, base = _inputs(batch, length, spread)
    want = _hc_split_sinkhorn_ops(mixes, scale, base, HC, ITERS, EPS)
    got = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    mx.eval(want, got)
    for name, a, b in zip(("pre", "post", "comb"), want, got):
        assert a.shape == b.shape, name
        assert float(mx.abs(a - b).max()) < 1e-5, name


@requires_kernel
def test_comb_columns_are_normalized():
    """The last sinkhorn step is a column normalization, so columns sum to 1.

    Rows only approach 1 -- three iterations do not fully converge -- so the
    row check is that the kernel is off by the same amount as the op path, not
    that it is close to 1.
    """
    mixes, scale, base = _inputs(1, 2, seed=3)
    _, _, kernel_comb = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    _, _, ops_comb = _hc_split_sinkhorn_ops(mixes, scale, base, HC, ITERS, EPS)
    mx.eval(kernel_comb, ops_comb)

    assert float(mx.abs(mx.sum(kernel_comb, axis=-2) - 1.0).max()) < 1e-4

    kernel_rows = mx.abs(mx.sum(kernel_comb, axis=-1) - 1.0)
    ops_rows = mx.abs(mx.sum(ops_comb, axis=-1) - 1.0)
    assert float(mx.abs(kernel_rows - ops_rows).max()) < 1e-5


def test_falls_back_when_hc_mult_is_not_four():
    """The kernel vectorizes comb over float4, so other widths take the ops."""
    mx.random.seed(1)
    hc = 2
    mix = (2 + hc) * hc
    mixes = mx.random.normal((1, 1, mix))
    scale = mx.ones((3,))
    base = mx.zeros((mix,))
    mx.eval(mixes, scale, base)
    got = hc_split_sinkhorn(mixes, scale, base, hc, ITERS, EPS)
    want = _hc_split_sinkhorn_ops(mixes, scale, base, hc, ITERS, EPS)
    mx.eval(got, want)
    for a, b in zip(want, got):
        assert bool(mx.array_equal(a, b).item())


@requires_kernel
def test_rows_are_independent():
    """Each row owns a threadgroup; a batched call must equal per-row calls."""
    mixes, scale, base = _inputs(1, 5, seed=7)
    batched = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
    mx.eval(batched)
    for i in range(mixes.shape[1]):
        single = hc_split_sinkhorn(mixes[:, i : i + 1], scale, base, HC, ITERS, EPS)
        mx.eval(single)
        for a, b in zip(batched, single):
            assert float(mx.abs(a[:, i : i + 1] - b).max()) < 1e-6
