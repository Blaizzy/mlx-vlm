"""Speculative decoding must honour --kv-bits, and work on quantized state.

Single-row MTP used to build its prompt cache with
``cache.make_prompt_cache``, bypassing the quantized cache that ``--kv-bits``
asks for: a 32K context then cost ~8.4GB of fp16 KV where 4-bit TurboQuant
needs ~2.1GB. Routing it through ``make_cache`` exposed a second gap --
target verification narrows the cache per draft token, which the quantized
state proxy did not support.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.speculative.utils import make_speculative_prompt_cache
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


class TestPromptCacheHonoursKvBits:
    """Every speculative configuration must go through ``make_cache``."""

    @pytest.mark.parametrize(
        "draft_kind,batch_size", [("mtp", 1), ("mtp", 2), (None, 1), ("eagle3", 1)]
    )
    def test_uses_supplied_make_cache(self, draft_kind, batch_size):
        sentinel = object()
        calls = []

        def make_cache(lm, left_padding):
            calls.append((lm, left_padding))
            return sentinel

        left_padding = [0] * batch_size
        result = make_speculative_prompt_cache(
            "lm",
            draft_kind=draft_kind,
            batch_size=batch_size,
            left_padding=left_padding,
            make_cache=make_cache,
        )

        # Single-row MTP used to return an unquantized cache built elsewhere.
        assert result is sentinel
        assert calls == [("lm", left_padding)]


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
