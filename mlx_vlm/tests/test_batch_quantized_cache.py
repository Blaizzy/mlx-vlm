"""Tests for BatchQuantizedKVCache — batch-aware quantized KV cache."""

import mlx.core as mx
import pytest

from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache

B, H, D = 2, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32
BITS = 8


def _rand_kv(batch, seq_len):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, H, seq_len, D))
    v = mx.random.normal((batch, H, seq_len, D))
    return k, v


class TestUpdateAndFetch:
    def test_basic_insert(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 5)
        qk, qv = cache.update_and_fetch(k, v)
        # Quantized state is a tuple of 3 arrays
        assert len(qk) == 3
        assert len(qv) == 3
        # Sequence dimension should match
        assert qk[0].shape[2] == 5
        assert cache._idx == 5

    def test_incremental_insert(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(B, 3)
        cache.update_and_fetch(k1, v1)
        k2, v2 = _rand_kv(B, 1)
        qk, qv = cache.update_and_fetch(k2, v2)
        assert qk[0].shape[2] == 4
        assert cache._idx == 4

    def test_offset_tracks_per_sequence(self):
        cache = BatchQuantizedKVCache([2, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 5)
        cache.update_and_fetch(k, v)
        offsets = cache.offset.tolist()
        # offset starts at [-2, 0] and adds 5
        assert offsets == [3, 5]


class TestFilter:
    def test_filter_keeps_correct_sequences(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(3, 4)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        cache.filter(mx.array([0, 2], mx.int32))
        # Should have 2 sequences left
        assert cache.keys[0].shape[0] == 2
        assert cache.offset.shape[0] == 2

    def test_filter_removes_common_left_padding(self):
        cache = BatchQuantizedKVCache([3, 1], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 6)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        # Keep only second sequence (left_padding=1)
        cache.filter(mx.array([1], mx.int32))
        # min left_padding=1, so it should shift left by 1
        assert cache.left_padding.tolist() == [0]
        assert cache._idx == 5  # 6 - 1

    def test_filter_single_sequence(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(3, 2)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        cache.filter(mx.array([1], mx.int32))
        assert cache.keys[0].shape[0] == 1


class TestExtend:
    def test_extend_concatenates_batches(self):
        c1 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(2, 4)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 4)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        assert c1.keys[0].shape[0] == 3
        assert c1.offset.shape[0] == 3
        assert c1.left_padding.shape[0] == 3

    def test_extend_handles_different_lengths(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(1, 8)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 3)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        # max_idx should be 8, the shorter one gets right-padded
        assert c1._idx == 8
        assert c1.keys[0].shape[0] == 2
        assert c1.left_padding.shape[0] == 2

    def test_extend_empty_into_populated(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(1, 4)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        c1.extend(c2)
        # Should still have 2 entries (1 populated + 1 empty offset)
        assert c1.offset.shape[0] == 2

    def test_extend_into_empty(self):
        c1 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 4)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)

        c1.extend(c2)
        assert c1._idx == 4
        assert c1.keys is not None


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


class TestMakeCache:
    """Test that _make_cache creates BatchQuantizedKVCache when kv_bits is set."""

    def test_make_cache_with_kv_bits(self):
        from mlx_vlm.generate import _make_cache
        from mlx_vlm.models.cache import BatchQuantizedKVCache as BQKV

        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer() for _ in range(4)]

        caches = _make_cache(FakeModel(), [0, 0], kv_bits=8, kv_group_size=64)
        # All but last should be quantized (model has >2 layers)
        assert isinstance(caches[0], BQKV)
        assert isinstance(caches[1], BQKV)
        assert isinstance(caches[2], BQKV)
        # Last layer should be unquantized
        assert isinstance(caches[3], BatchKVCache)

    def test_make_cache_without_kv_bits(self):
        from mlx_vlm.generate import _make_cache

        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer() for _ in range(4)]

        caches = _make_cache(FakeModel(), [0, 0])
        for c in caches:
            assert isinstance(c, BatchKVCache)


class TestBatchGeneratorIntegration:
    """Test that BatchGenerator accepts and propagates kv_bits."""

    def test_batch_generator_accepts_kv_params(self):
        from unittest.mock import Mock

        from mlx_vlm.generate import BatchGenerator

        model = Mock()
        model.layers = [Mock() for _ in range(2)]
        proc = Mock()
        proc.tokenizer = Mock()
        proc.tokenizer.stopping_criteria = Mock()
        proc.tokenizer.stopping_criteria.add_eos_token_ids = Mock()

        gen = BatchGenerator(
            model, proc, kv_bits=4, kv_group_size=64, quantized_kv_start=100
        )
        assert gen.kv_bits == 4
        assert gen.kv_group_size == 64
        assert gen.quantized_kv_start == 100

    def test_batch_generator_default_no_quantization(self):
        from unittest.mock import Mock

        from mlx_vlm.generate import BatchGenerator

        model = Mock()
        model.layers = [Mock() for _ in range(2)]
        proc = Mock()
        proc.tokenizer = Mock()
        proc.tokenizer.stopping_criteria = Mock()
        proc.tokenizer.stopping_criteria.add_eos_token_ids = Mock()

        gen = BatchGenerator(model, proc)
        assert gen.kv_bits is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
