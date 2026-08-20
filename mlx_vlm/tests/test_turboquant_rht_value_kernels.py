"""RHT-path equivalence for the L=1 value Metal kernels.

The single-token value kernels (``_metal_mse_weighted_sum`` and friends)
compute the weighted value sum in the codec's *rotated* space. They used to
hard-code ``matmul(weighted_rot, rotation)`` to undo that rotation, which is
only correct for the dense-rotation codec — so #1244 disabled them whenever
the codec uses the Randomized Hadamard Transform (``use_rht``), falling back
to the slower einsum path for every RHT decode.

The wrappers now apply the codec's RHT-aware inverse (via ``signs``), so the
fast kernel path is valid under RHT too. These tests pin that the kernel and
einsum paths agree, and that RHT decode actually takes the kernel path.
"""

import mlx.core as mx
import pytest

import mlx_vlm.turboquant as tq
from mlx_vlm.turboquant import _TurboQuantMSECodec


def _codec_and_state(dim: int, bits: int, n_heads: int, n_tokens: int, seed: int):
    mx.random.seed(seed)
    values = mx.random.normal((1, n_heads, n_tokens, dim))
    codec = _TurboQuantMSECodec(dim, bits, seed=seed)
    return codec, codec.quantize(values)


def _dequant_weighted_sum(codec, state, weights):
    """Math ground truth: weighted sum over the codec's own dequantized values."""
    deq = codec.dequantize(state)  # (1, H, T, D)
    return mx.einsum("bhmlt,bhtd->bhmld", weights, deq)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("n_repeats", [1, 4])
def test_rht_weighted_sum_matches_einsum_and_dequant(dim, bits, n_repeats, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    n_heads, n_tokens = 2, 24
    codec, state = _codec_and_state(dim, bits, n_heads, n_tokens, seed=0)
    # power-of-2 dim -> the RHT path that #1244 disabled for these kernels
    assert codec.use_rht is True

    weights = mx.softmax(
        mx.random.normal((1, n_heads, n_repeats, 1, n_tokens)), axis=-1
    )

    kernel_out = codec.weighted_sum(weights, state)  # Metal -> RHT kernel fast path
    truth = _dequant_weighted_sum(codec, state, weights)

    # Force the einsum fallback by hiding Metal, then compare paths.
    monkeypatch.setattr(tq, "_metal_available", lambda: False)
    einsum_out = codec.weighted_sum(weights, state)

    assert kernel_out.shape == einsum_out.shape == truth.shape
    assert mx.max(mx.abs(kernel_out - einsum_out)).item() < 1e-4
    assert mx.max(mx.abs(kernel_out - truth)).item() < 1e-3


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bits", [3, 4])
def test_rht_weighted_sum_stats_matches_einsum(dim, bits, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    n_heads, n_repeats, n_tokens = 2, 4, 24
    codec, state = _codec_and_state(dim, bits, n_heads, n_tokens, seed=1)
    assert codec.use_rht is True

    scores = mx.random.normal((1, n_heads, n_repeats, 1, n_tokens))

    out_k, denom_k, max_k = codec.weighted_sum_stats_from_scores(scores, state)

    monkeypatch.setattr(tq, "_metal_available", lambda: False)
    out_e, denom_e, max_e = codec.weighted_sum_stats_from_scores(scores, state)

    assert mx.max(mx.abs(out_k - out_e)).item() < 1e-4
    assert mx.max(mx.abs(denom_k - denom_e)).item() < 1e-4
    assert mx.max(mx.abs(max_k - max_e)).item() < 1e-4


def test_rht_weighted_sum_takes_kernel_path_not_unpack(monkeypatch):
    """Under RHT the fast kernel must run — never the einsum fallback (which unpacks).

    Regression guard for the perf goal: before this fix the RHT codec skipped
    the kernel and fell through to einsum, calling ``_unpack_lowbit``.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    n_heads, n_repeats, n_tokens, dim = 2, 4, 24, 64
    codec, state = _codec_and_state(dim, 4, n_heads, n_tokens, seed=2)
    assert codec.use_rht is True

    weights = mx.softmax(
        mx.random.normal((1, n_heads, n_repeats, 1, n_tokens)), axis=-1
    )

    def fail(*args, **kwargs):
        raise AssertionError("RHT weighted_sum should use the Metal kernel, not unpack")

    monkeypatch.setattr(tq, "_unpack_lowbit", fail)
    output = codec.weighted_sum(weights, state)
    mx.eval(output)
    assert output.shape == (1, n_heads, n_repeats, 1, dim)
