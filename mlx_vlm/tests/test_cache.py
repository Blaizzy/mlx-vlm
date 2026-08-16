import mlx.core as mx
import pytest

from mlx_vlm.models.cache import (
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)


def _make_kv_cache(batch_size=1, length=3):
    cache = KVCache()
    keys = mx.arange(batch_size * 2 * length * 4).reshape(batch_size, 2, length, 4)
    values = keys + 100
    cache.update_and_fetch(keys, values)
    return cache, keys, values


def test_kv_cache_extracts_one_active_row():
    cache, keys, values = _make_kv_cache(batch_size=2)

    extracted = cache.extract(1)

    assert extracted.offset == 3
    assert extracted.keys.shape == (1, 2, 3, 4)
    assert extracted.values.shape == (1, 2, 3, 4)
    assert mx.array_equal(extracted.keys, keys[1:2]).item()
    assert mx.array_equal(extracted.values, values[1:2]).item()


def test_cache_list_can_extract_an_already_extracted_kv_cache():
    first, _, _ = _make_kv_cache()
    second, _, _ = _make_kv_cache()
    batched = CacheList.merge([CacheList(first), CacheList(second)])

    extracted = batched.extract(0)
    extracted_again = extracted.extract(0)

    assert isinstance(extracted_again[0], KVCache)
    assert extracted_again[0].offset == first.offset
    assert mx.array_equal(extracted_again[0].keys, first.state[0]).item()
    assert mx.array_equal(extracted_again[0].values, first.state[1]).item()


def test_kv_cache_extract_validates_row_index():
    cache, _, _ = _make_kv_cache(batch_size=2)

    assert mx.array_equal(cache.extract(-1).keys, cache.extract(1).keys).item()
    with pytest.raises(IndexError):
        cache.extract(2)
    with pytest.raises(IndexError):
        cache.extract(-3)


def test_empty_kv_cache_extracts_as_empty():
    extracted = KVCache().extract(0)

    assert extracted.empty()
    assert extracted.offset == 0


def test_batch_rotating_merge_skips_zero_length_backing_storage():
    def make_cache(length):
        cache = RotatingKVCache(max_size=32)
        cache.keys = mx.zeros((1, 2, 24, 4))
        cache.values = mx.ones((1, 2, 24, 4))
        cache._idx = 24
        cache.offset = length
        return cache

    merged = BatchRotatingKVCache.merge([make_cache(0), make_cache(24)])
    mx.eval(merged.keys, merged.values)

    assert merged.keys.shape == (2, 2, 24, 4)
    assert merged.values.shape == (2, 2, 24, 4)
    assert merged.offset.tolist() == [0, 24]
    assert mx.all(merged.keys[0] == 0).item()
    assert mx.all(merged.values[0] == 0).item()
