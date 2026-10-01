"""Numerical and boundary checks for fused window + compressed attention."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
from mlx_vlm.models.deepseek_v41.sparse_attention import sparse_attention

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


def inputs(
    batch=1,
    heads=4,
    length=17,
    dim=512,
    history=7,
    topk=31,
    dtype=mx.bfloat16,
    window_size=16,
    pool_length=97,
):
    mx.random.seed(71)
    q = (
        mx.random.normal((batch, length, heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    window = mx.random.normal((batch, length + history, dim)).astype(dtype)
    pool = mx.random.normal((batch, pool_length, dim)).astype(dtype)
    indices = mx.random.randint(0, pool_length, (batch, length, topk))
    indices = mx.where(mx.arange(topk) % 5 == 0, -1, indices)
    positions = mx.arange(length + history)
    ends = mx.arange(length) + history
    mask = ((positions <= ends[:, None]) & (positions > ends[:, None] - window_size))[
        None, None
    ]
    sinks = mx.linspace(-4, 6, heads)
    return q, window, pool, indices, mask, sinks, dim**-0.5, window_size


def reference(args):
    q, window, pool, indices, mask, sinks, scale, _ = args
    return _sparse_pooled_attention(
        q.astype(mx.float32),
        window[:, None],
        pool,
        indices,
        mask,
        (indices != -1)[:, None],
        scale,
        sinks.astype(mx.float32),
    ).astype(q.dtype)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "batch,length,heads,dim,topk,history",
    [
        (1, 1, 64, 512, 512, 127),
        (4, 1, 64, 512, 512, 127),
        (1, 17, 4, 512, 31, 0),
        (4, 33, 8, 64, 15, 7),
        (1, 129, 4, 512, 512, 127),
        (1, 17, 64, 512, 33, 0),
        (1, 129, 8, 512, 512, 127),
    ],
)
def test_matches_reference(dtype, batch, length, heads, dim, topk, history):
    args = inputs(batch, heads, length, dim, history, topk, dtype, 128)
    actual, expected = sparse_attention(*args), reference(args)
    assert actual is not None
    mx.eval(actual, expected)
    # FP32 reduction ordering can move the final half-precision rounding by one ULP.
    tolerance = 0.004 if dtype == mx.bfloat16 else 0.0005
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()


@pytest.mark.parametrize("sink", [-1000.0, 0.0, 1000.0])
@pytest.mark.parametrize("heads", [2, 8])
def test_masked_rows_and_attention_sinks(sink, heads):
    args = list(inputs(batch=2, heads=heads, length=3, dim=64))
    args[3] = mx.full(args[3].shape, -1, mx.int64)
    args[4] = mx.zeros((2, 1, 3, args[1].shape[1]), mx.bool_)
    args[5] = mx.full((heads,), sink)
    out = sparse_attention(*args)
    assert mx.all(mx.isfinite(out)).item()
    assert mx.all(out == 0).item()


@pytest.mark.parametrize("heads", [3, 8])
def test_float64_oracle_with_duplicate_indices_and_batch_mask(heads):
    args = list(inputs(batch=2, heads=heads, length=3, dim=64, topk=4))
    args[3] = mx.broadcast_to(mx.array([3, 3, -1, 5]), (2, 3, 4))
    args[4] = mx.concatenate([args[4], mx.zeros_like(args[4])], axis=0)
    q, window, pool, indices, mask, sinks = [
        np.array(a.astype(mx.float32)) for a in args[:6]
    ]
    expected = np.zeros_like(q, dtype=np.float64)
    for b in range(2):
        for h in range(heads):
            for t in range(3):
                keys = np.concatenate(
                    [
                        window[b, mask[b, 0, t].astype(bool)],
                        pool[b, indices[b, t][indices[b, t] >= 0].astype(int)],
                    ]
                )
                scores = keys.astype(np.float64) @ (
                    q[b, h, t].astype(np.float64) * args[6]
                )
                maximum = max(scores.max(), sinks[h])
                weights = np.exp(scores - maximum)
                expected[b, h, t] = (
                    weights @ keys / (weights.sum() + np.exp(sinks[h] - maximum))
                )
    actual = sparse_attention(*args)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, atol=0.004, rtol=0.004
    )


def test_empty_pool_and_no_selected_keys():
    args = list(inputs(length=3, topk=0))
    args[2] = args[2][:, :0]
    actual = sparse_attention(*args)
    q, window, _, _, mask, sinks, scale, _ = args
    scores = (q.astype(mx.float32) * scale) @ window[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    scores = mx.where(mask, scores, -mx.inf)
    probabilities = mx.softmax(
        mx.concatenate(
            [
                scores,
                mx.broadcast_to(sinks[None, :, None, None], (*scores.shape[:-1], 1)),
            ],
            -1,
        ),
        -1,
    )
    expected = probabilities[..., :-1] @ window[:, None].astype(mx.float32)
    assert mx.allclose(
        actual.astype(mx.float32), expected, atol=0.004, rtol=0.004
    ).item()


def test_unsupported_inputs_return_none():
    args = list(inputs())
    args[0] = args[0].astype(mx.float32)
    assert sparse_attention(*args) is None
    args = list(inputs())
    args[4] = args[4].astype(mx.float32)
    assert sparse_attention(*args) is None
    args = list(inputs(dim=48))
    assert sparse_attention(*args) is None


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_split_decode_matches_original_reduction_exactly(dtype, batch, masked):
    args = list(inputs(batch, 8, 1, 512, 127, 512, dtype, 128))
    # Include repeated keys, invalid routes, batch-specific masks, and sink extremes.
    args[3] = mx.broadcast_to(mx.array([3, 3, -1, 5] * 128), (batch, 1, 512))
    args[4] = mx.broadcast_to(args[4], (batch, 1, 1, 128))
    args[4] = args[4] & (mx.arange(batch)[:, None, None, None] % 2 == 0)
    args[5] = mx.array([-1000, -10, -1, 0, 1, 10, 100, 1000], mx.float32)
    if masked:
        args[3] = mx.full(args[3].shape, -1, mx.int32)
        args[4] = mx.zeros_like(args[4])
    actual = sparse_attention(*args)
    # Batch 1 retains the original one-threadgroup decode kernel.
    expected = mx.concatenate(
        [
            sparse_attention(*(a[row : row + 1] for a in args[:5]), *args[5:])
            for row in range(batch)
        ],
        axis=0,
    )
    assert mx.array_equal(actual, expected).item()
