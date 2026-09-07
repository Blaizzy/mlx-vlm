import mlx.core as mx
import pytest

from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    PoolingCache,
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


def test_arrays_cache_advance_matches_decremented_values():
    cache = ArraysCache(1, left_padding=[3, 1])
    cache.prepare(lengths=[10, 7])
    cache.advance(1)
    cache.advance(2)

    assert cache.left_padding.tolist() == [0, -2]
    assert cache.lengths.tolist() == [7, 4]

    mask = cache.make_mask(4)
    assert mask.tolist() == [[p >= lp for p in range(4)] for lp in (0, -2)]

    cache.filter([1, 0])
    assert cache.left_padding.tolist() == [-2, 0]
    assert cache.lengths.tolist() == [4, 7]

    cache.finalize()
    assert cache.left_padding is None and cache.lengths is None


def test_arrays_cache_advance_does_not_accumulate_buffers():
    cache = ArraysCache(1, left_padding=[0, 0])
    cache.prepare(lengths=[8, 8])

    mx.clear_cache()
    base = mx.get_active_memory()
    for _ in range(20000):
        cache.advance(1)
    delta = mx.get_active_memory() - base

    assert delta < 4096, f"advance leaked {delta} bytes over 20k steps"
    assert cache.lengths.tolist() == [8 - 20000, 8 - 20000]


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


def _advance_pooling_cache(cache, values):
    gate = values + 1000
    ready, _, _ = cache.accumulate_windows(values, gate, offset=0)
    if ready.shape[1]:
        pooled = ready.reshape(ready.shape[0], -1, cache.ratio, ready.shape[-1]).sum(
            axis=2
        )
    else:
        pooled = mx.zeros((ready.shape[0], 0, ready.shape[-1]), dtype=ready.dtype)
    cache.update_and_fetch(pooled)


@pytest.mark.parametrize("initial_length", range(4))
@pytest.mark.parametrize("keep", range(5))
def test_pooling_cache_speculative_commit_matches_prefix_replay(initial_length, keep):
    actual = PoolingCache(ratio=4)
    reference = PoolingCache(ratio=4)
    initial = mx.arange(initial_length, dtype=mx.float32).reshape(1, -1, 1)
    block = mx.arange(10, 14, dtype=mx.float32).reshape(1, 4, 1)
    if initial_length:
        _advance_pooling_cache(actual, initial)
        _advance_pooling_cache(reference, initial)

    generation = actual.start_speculation(block.shape[1])
    _advance_pooling_cache(actual, block)
    if keep:
        _advance_pooling_cache(reference, block[:, :keep])
    actual.commit_speculation(keep, generation)
    mx.eval(actual.state, reference.state)

    assert actual.remainder == reference.remainder
    assert actual.offset == reference.offset
    if actual.remainder:
        assert mx.array_equal(
            actual.buf_kv[:, : actual.remainder],
            reference.buf_kv[:, : reference.remainder],
        ).item()
        assert mx.array_equal(
            actual.buf_gate[:, : actual.remainder],
            reference.buf_gate[:, : reference.remainder],
        ).item()
    if reference.pooled is None:
        assert actual.pooled is None
    else:
        assert mx.array_equal(actual.pooled, reference.pooled).item()


def test_batch_pooling_cache_speculative_commit_matches_ragged_prefixes():
    batch, ratio = 4, 4
    padding = [3, 2, 1, 0]
    actual = BatchPoolingCache(ratio=ratio, left_padding=padding)
    references = [PoolingCache(ratio=ratio) for _ in range(batch)]
    initial = mx.arange(batch * 3, dtype=mx.float32).reshape(batch, 3, 1)
    block = mx.arange(100, 100 + batch * 4, dtype=mx.float32).reshape(batch, 4, 1)
    keep = [1, 2, 3, 4]

    _advance_pooling_cache(actual, initial)
    for row, left_padding in enumerate(padding):
        if left_padding < initial.shape[1]:
            _advance_pooling_cache(
                references[row], initial[row : row + 1, left_padding:]
            )

    generation = actual.start_speculation(block.shape[1])
    _advance_pooling_cache(actual, block)
    actual.commit_speculation(keep, generation)
    for row, length in enumerate(keep):
        _advance_pooling_cache(references[row], block[row : row + 1, :length])

    mx.eval(actual.state, [reference.state for reference in references])
    assert actual.remainder == [reference.remainder for reference in references]
    assert actual._pool_lengths == [reference.offset for reference in references]
    assert actual._processed == [3 - padding[row] + keep[row] for row in range(batch)]
    for row, reference in enumerate(references):
        remainder = reference.remainder
        assert mx.array_equal(
            actual.buf_kv[row : row + 1, :remainder],
            reference.buf_kv[:, :remainder],
        ).item()
        pooled_length = reference.offset
        if pooled_length:
            assert mx.array_equal(
                actual.pooled[row : row + 1, :pooled_length],
                reference.pooled,
            ).item()


def test_empty_kv_cache_extracts_as_empty():
    extracted = KVCache().extract(0)

    assert extracted.empty()
    assert extracted.offset == 0


def test_empty_batch_kv_cache_ignores_unapplied_right_padding():
    cache = BatchKVCache([0, 0])

    cache.prepare(right_padding=[0, 1])
    cache.finalize()

    assert cache.empty()
    assert cache.offset.tolist() == [0, 0]
    assert cache.left_padding.tolist() == [0, 0]
    assert cache._right_padding is None


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


def test_chunked_kv_cache_trims_on_valid_length_not_buffer_width():
    # Regression test for maybe_trim_front: update_and_fetch pads the backing
    # buffer up to a multiple of ``step``, so trimming on ``keys.shape[2]``
    # discarded up to ``step - 1`` live tokens per trim and left the attention
    # window narrower than ``chunk_size``.
    chunk_size, n_tokens = 8, 14
    original_step = ChunkedKVCache.step
    ChunkedKVCache.step = 4
    try:
        cache = ChunkedKVCache(chunk_size)
        window = []
        for token in range(n_tokens):
            # llama4/language.py trims before every update on chunked layers.
            cache.maybe_trim_front()
            kv = mx.full((1, 1, 1, 2), float(token))
            keys, _ = cache.update_and_fetch(kv, kv)
            window = [int(k) for k in keys[0, 0, :, 0]]
    finally:
        ChunkedKVCache.step = original_step

    # The window must never fall below chunk_size once that many tokens exist,
    # and must be the contiguous run ending at the newest token.
    assert len(window) >= chunk_size
    assert window[-1] == n_tokens - 1
    assert window == list(range(window[0], window[0] + len(window)))
    assert cache.start_position <= cache.offset
