import mlx.core as mx
import pytest

from mlx_vlm.models.cache import (
    ArraysCache,
    BatchPoolingCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    UnboundedKVCache,
    make_prompt_cache,
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


def test_batch_pooling_cache_skips_left_padding_across_chunks():
    cache = BatchPoolingCache(ratio=4, left_padding=[5, 0])
    kv = mx.array(
        [
            [[90], [91], [92], [93], [94], [1], [2], [3], [4]],
            [[10], [11], [12], [13], [14], [15], [16], [17], [18]],
        ],
        dtype=mx.float32,
    )
    gate = kv + 100
    offsets = [mx.array([-5, 0]), mx.array([-2, 3]), mx.array([1, 6])]

    outputs = []
    for chunk, offset in zip(range(0, 9, 3), offsets):
        outputs.append(
            cache.accumulate_windows(
                kv[:, chunk : chunk + 3],
                gate[:, chunk : chunk + 3],
                offset,
            )
        )

    ready_kv, ready_gate, pool_base = outputs[-1]
    assert ready_kv[:, :, 0].tolist() == [
        [1.0, 2.0, 3.0, 4.0],
        [14.0, 15.0, 16.0, 17.0],
    ]
    assert ready_gate[:, :, 0].tolist() == [
        [101.0, 102.0, 103.0, 104.0],
        [114.0, 115.0, 116.0, 117.0],
    ]
    assert pool_base.tolist() == [0, 4]
    assert cache.left_padding == [0, 0]
    assert cache._processed == [4, 9]
    assert cache.remainder == [0, 1]
    assert cache.buf_kv[:, :1, 0].tolist() == [[0.0], [18.0]]


def test_batch_pooling_cache_tracks_left_padding_through_batch_operations():
    cache = BatchPoolingCache(ratio=4, left_padding=[3, 1])
    other = BatchPoolingCache(ratio=4, left_padding=[2])

    cache.extend(other)
    cache.filter([2, 0])

    assert cache.left_padding == [2, 3]

    restored = BatchPoolingCache.from_state(cache.state, cache.meta_state)
    assert restored.left_padding == [2, 3]
    restored.prepare(left_padding=[1, 2])
    assert restored.left_padding == [3, 5]


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


class _ZeroArg:
    """The common shape: a uniform KV cache, built with no arguments."""

    def make_cache(self):
        return [KVCache(), KVCache()]


class _Hybrid:
    """Recurrent state interleaved with attention, as the hybrid models build."""

    def make_cache(self):
        return [ArraysCache(size=2), KVCache(), ArraysCache(size=2)]


class _Nested:
    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(size=2))]


class _AcceptsMaxSize:
    def make_cache(self, max_size=None):
        self.seen = max_size
        return [RotatingKVCache(max_size=max_size or 999)]


class _SwallowsKwargs:
    """A bare ``**kwargs`` must not count as accepting the bound."""

    def make_cache(self, **kwargs):
        self.seen = kwargs
        return [KVCache()]


class _AlreadyTighter:
    def make_cache(self):
        return [RotatingKVCache(max_size=16)]


class _NotBoundable:
    def make_cache(self):
        return [ArraysCache(size=2), ArraysCache(size=2)]


def test_max_kv_size_bounds_a_zero_argument_make_cache():
    cache = make_prompt_cache(_ZeroArg(), max_kv_size=64)
    assert [type(c) for c in cache] == [RotatingKVCache, RotatingKVCache]
    assert {c.max_size for c in cache} == {64}


def test_max_kv_size_leaves_fixed_size_state_alone():
    cache = make_prompt_cache(_Hybrid(), max_kv_size=64)
    assert [type(c).__name__ for c in cache] == [
        "ArraysCache",
        "RotatingKVCache",
        "ArraysCache",
    ]
    assert cache[1].max_size == 64


def test_max_kv_size_recurses_into_cache_list():
    cache = make_prompt_cache(_Nested(), max_kv_size=64)
    inner = list(cache[0].caches)
    assert isinstance(inner[0], RotatingKVCache) and inner[0].max_size == 64
    assert type(inner[1]).__name__ == "ArraysCache"


def test_max_kv_size_is_passed_to_a_make_cache_that_accepts_it():
    model = _AcceptsMaxSize()
    cache = make_prompt_cache(model, max_kv_size=64)
    assert model.seen == 64
    assert cache[0].max_size == 64


def test_bare_kwargs_does_not_count_as_accepting_the_bound():
    model = _SwallowsKwargs()
    cache = make_prompt_cache(model, max_kv_size=64)
    assert not getattr(model, "seen", None)
    assert isinstance(cache[0], RotatingKVCache) and cache[0].max_size == 64


def test_a_tighter_existing_bound_is_kept():
    cache = make_prompt_cache(_AlreadyTighter(), max_kv_size=64)
    assert cache[0].max_size == 16


def test_unboundable_cache_raises_instead_of_ignoring_the_request():
    with pytest.raises(UnboundedKVCache, match="max_kv_size=64"):
        make_prompt_cache(_NotBoundable(), max_kv_size=64)


def test_no_bound_requested_leaves_the_model_cache_untouched():
    assert [type(c) for c in make_prompt_cache(_ZeroArg())] == [KVCache, KVCache]
    assert [type(c).__name__ for c in make_prompt_cache(_Hybrid())] == [
        "ArraysCache",
        "KVCache",
        "ArraysCache",
    ]
