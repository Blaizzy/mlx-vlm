import mlx.core as mx
import pytest

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.turboquant import BatchTurboQuantKVCache, _state_length

H, D = 4, 64
BITS = 4
SCALE = D**-0.5


def _filled(seq_len, left_padding=(0,)):
    cache = BatchTurboQuantKVCache(list(left_padding), bits=BITS)
    batch = len(left_padding)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, H, seq_len, D)),
        mx.random.normal((batch, H, seq_len, D)),
    )
    return cache, keys, values


class TestQuantizedStateSlicing:
    """Target verification narrows the cache one draft token at a time."""

    def test_slice_reports_narrowed_length(self):
        _, keys, _ = _filled(300)
        for n in (1, 128, 300):
            narrowed = keys[:, :, :n, :]
            assert narrowed.shape[2] == n
            assert _state_length(narrowed._state) == n

    def test_slice_stays_quantized(self):
        # A slice that dequantized would defeat the point of --kv-bits.
        _, keys, _ = _filled(64)
        assert type(keys[:, :, :32, :]) is type(keys)

    def test_sliced_state_matches_dequantized_prefix(self):
        cache, keys, values = _filled(300)
        mx.eval(cache.keys, cache.values)
        queries = mx.random.normal((1, H, 1, D))

        prefix_keys, prefix_values = keys[:, :, :128, :], values[:, :, :128, :]
        out = scaled_dot_product_attention(
            queries, prefix_keys, prefix_values, cache=cache, scale=SCALE, mask=None
        )
        dq_k, dq_v = cache.dequantize(prefix_keys, prefix_values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=None,
        )
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()

    @pytest.mark.parametrize(
        "key",
        [
            (slice(None), slice(None), slice(5, 10), slice(None)),  # offset start
            (slice(None), 0, slice(None, 10), slice(None)),  # indexes a head
            (slice(None), slice(None), slice(None, 10, 2), slice(None)),  # strided
        ],
    )
    def test_rejects_unsupported_indexing(self, key):
        _, keys, _ = _filled(64)
        with pytest.raises(TypeError):
            keys[key]
