"""Tests for BatchQuantizedKVCache — batch-aware quantized KV cache."""

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import BatchQuantizedKVCache

B, H, D = 2, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32
BITS = 8


def _rand_kv(batch, seq_len):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, H, seq_len, D))
    v = mx.random.normal((batch, H, seq_len, D))
    return k, v


class TestExtend:
    def test_extend_handles_filtered_non_step_aligned_capacity(self):
        c1 = BatchQuantizedKVCache([7, 7], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(2, 512)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        # Filtering rows can trim common left padding and leave a backing
        # sequence length that is no longer aligned to the allocation step.
        c1.filter(mx.array([0], mx.int32))
        mx.eval(c1.keys)
        assert c1.keys[0].shape[-2] == 505
        assert c1._idx == 505

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 500)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)
        assert c2.keys[0].shape[-2] == 512
        assert c2._idx == 500

        c1.extend(c2)
        mx.eval(c1.keys)

        assert c1.keys[0].shape[0] == 2
        assert c1.keys[0].shape[-2] == 512
        assert c1._idx == 505
        assert c1.left_padding.tolist() == [0, 5]


class TestState:
    def test_state_roundtrip(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        state = cache.state
        assert len(state) == 4  # keys, values, offset, left_padding

        cache2 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        cache2.state = state
        assert cache2._idx == 4

    def test_empty_state(self):
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        state = cache.state
        assert state[0] is None
        assert state[1] is None


class TestPrepareFinalize:
    """Multi-row right-pad lifecycle parity with BatchKVCache (#1567 / #1562)."""

    def test_finalize_noop_without_prepare(self):
        cache = BatchQuantizedKVCache([1, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        before = cache.left_padding.tolist()
        cache.finalize()
        assert cache.left_padding.tolist() == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
