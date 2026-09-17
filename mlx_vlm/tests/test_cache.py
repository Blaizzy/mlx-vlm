"""Cache lifecycle, quantization, recurrence, and batched attention masks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate import generate_step
from mlx_vlm.models.base import (
    InputEmbeddingsFeatures,
    LanguageModelOutput,
    align_attention_mask_to_scores,
    quantized_scaled_dot_product_attention,
)
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    PoolingCache,
    RotatingKVCache,
    create_causal_mask,
)
from mlx_vlm.models.hy_v4.cache import HyV4KVCache


def _make_kv_cache(batch_size=1, length=3, factory=KVCache):
    cache = factory()
    keys = mx.arange(batch_size * 2 * length * 4).reshape(batch_size, 2, length, 4)
    values = keys + 100
    cache.update_and_fetch(keys, values)
    return cache, keys, values


@pytest.mark.parametrize(
    "factory",
    [KVCache, HyV4KVCache, lambda: BatchKVCache([0]), lambda: RotatingKVCache(8)],
)
def test_empty_kv_cache_state_can_be_evaluated_and_restored(factory):
    cache = factory()
    # Cross-attention layers can leave their cache empty during chunked prefill.
    mx.eval(cache.state)
    restored = factory()
    restored.state = cache.state
    restored.meta_state = cache.meta_state
    assert restored.empty()
    keys = mx.ones((1, 2, 3, 4))
    values = keys * 2
    actual_keys, actual_values = restored.update_and_fetch(keys, values)
    assert mx.array_equal(actual_keys, keys).item()
    assert mx.array_equal(actual_values, values).item()


@pytest.mark.parametrize("factory", [KVCache, HyV4KVCache])
def test_kv_cache_extracts_one_active_row(factory):
    cache, keys, values = _make_kv_cache(batch_size=2, factory=factory)

    extracted = cache.extract(1)

    assert type(extracted) is factory
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


@pytest.mark.parametrize("family", ["shared", "qwen"])
@pytest.mark.parametrize("use_kernel", [False, True])
@pytest.mark.parametrize("parts", [(4,), (1, 1, 1, 1), (1, 3), (2, 2)])
@pytest.mark.parametrize("retained", [[0, 1, 3, 4], [1] * 4, [4] * 4])
def test_temporal_recurrence_commits_exact_states(family, use_kernel, parts, retained):
    from mlx_vlm.models.gated_delta import gated_delta_update as shared_update
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update as qwen_update

    update = shared_update if family == "shared" else qwen_update
    mx.random.seed(2127)
    batch, length, heads, width = 4, 4, 2, 32
    q, k, v = [
        mx.random.normal((batch, length, heads, width)).astype(mx.bfloat16) * 0.1
        for _ in range(3)
    ]
    gate_shape = q.shape if family == "shared" else q.shape[:-1]
    a = mx.random.normal(gate_shape).astype(mx.bfloat16)
    b = mx.random.normal(q.shape[:-1]).astype(mx.bfloat16)
    weights_shape = (heads, 1) if family == "shared" else (heads,)
    decay, bias = mx.zeros(weights_shape), mx.zeros(weights_shape)
    kwargs = {"lower_bound": -5.0} if family == "shared" else {}
    initial = mx.random.normal((batch, heads, width, width)) * 0.01
    cache = ArraysCache(1)
    cache[0] = initial
    generation = cache.start_speculation(length)
    position = 0
    outputs = []
    for size in parts:
        output, _ = update(
            *(x[:, position : position + size] for x in (q, k, v, a, b)),
            decay,
            bias,
            cache=cache,
            cache_index=0,
            use_kernel=use_kernel,
            **kwargs,
        )
        outputs.append(output)
        position += size
        assert cache._recorded_length(0) == position
        assert cache.nbytes <= initial.nbytes * (length + 1)

    expected_outputs = []
    states = [initial]
    for position in range(length):
        output, state = update(
            *(x[:, position : position + 1] for x in (q, k, v, a, b)),
            decay,
            bias,
            state=states[-1],
            use_kernel=use_kernel,
            **kwargs,
        )
        expected_outputs.append(output)
        states.append(state)
    assert mx.array_equal(
        mx.concatenate(outputs, axis=1), mx.concatenate(expected_outputs, axis=1)
    ).item()
    cache.commit_speculation(retained, generation)
    expected = mx.concatenate(
        [states[keep][row : row + 1] for row, keep in enumerate(retained)]
    )
    assert mx.array_equal(cache[0], expected).item()
    assert cache.history_capacity == 0
    assert cache.nbytes == cache[0].nbytes


def test_temporal_cache_retention_is_bounded_and_abort_restores_state():
    cache = ArraysCache(1)
    initial = mx.ones((2, 3))
    cache[0] = initial
    requested = []

    def step(state, state_steps):
        requested.append(state_steps)
        return state, state + 1

    cache.update_recurrent(0, 1, step)
    assert requested == [None]
    assert cache.nbytes == initial.nbytes
    initial = cache[0]
    for _ in range(8):
        generation = cache.start_speculation(2)
        cache.update_recurrent(0, 1, step)
        with pytest.raises(RuntimeError, match="full window"):
            cache.commit_speculation(1, generation)
        cache.update_recurrent(0, 1, step)
        count = len(requested)
        with pytest.raises(ValueError, match="capacity"):
            cache.update_recurrent(0, 1, step)
        assert len(requested) == count
        assert cache.nbytes <= initial.nbytes * 3
        cache.abort_speculation(generation)
        assert cache[0] is initial
        assert cache.nbytes == initial.nbytes


@pytest.mark.parametrize("factory", [KVCache, HyV4KVCache])
def test_kv_cache_extract_validates_row_index(factory):
    cache, _, _ = _make_kv_cache(batch_size=2, factory=factory)

    assert mx.array_equal(cache.extract(-1).keys, cache.extract(1).keys).item()
    with pytest.raises(IndexError):
        cache.extract(2)
    with pytest.raises(IndexError):
        cache.extract(-3)


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


@pytest.mark.parametrize("incremental", [False, True])
def test_batch_pooling_cache_speculative_commit_matches_ragged_prefixes(incremental):
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
    if incremental:
        for index in range(block.shape[1]):
            _advance_pooling_cache(actual, block[:, index : index + 1])
    else:
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
            actual.buf_kv[row : row + 1, :remainder], reference.buf_kv[:, :remainder]
        ).item()
        pooled_length = reference.offset
        if pooled_length:
            assert mx.array_equal(
                actual.pooled[row : row + 1, :pooled_length], reference.pooled
            ).item()


@pytest.mark.parametrize("factory", [KVCache, HyV4KVCache])
def test_empty_kv_cache_extracts_as_empty(factory):
    cache = factory()
    for idx in (0, -1):
        extracted = cache.extract(idx)
        assert type(extracted) is factory
        assert extracted.empty()
        assert extracted.offset == 0
    with pytest.raises(IndexError):
        cache.extract(1)


def test_empty_batch_kv_cache_ignores_unapplied_right_padding():
    cache = BatchKVCache([0, 0])

    cache.prepare(right_padding=[0, 1])
    cache.finalize()

    assert cache.empty()
    assert cache.offset.tolist() == [0, 0]
    assert cache.left_padding.tolist() == [0, 0]
    assert cache._right_padding is None


@pytest.mark.parametrize("window", [4, 8, 16])
@pytest.mark.parametrize("prefix_length", [0, 3, 8, 20])
@pytest.mark.parametrize("parts", [(3, 1), (1, 1, 1, 1)])
@pytest.mark.parametrize("right_padded", [False, True])
def test_batch_rotating_masks_match_prefill_and_decode_layouts(
    window, prefix_length, parts, right_padded
):
    head_dim = 64

    def values(rows):
        array = mx.array(rows, dtype=mx.float32)[:, None, :, None]
        return mx.broadcast_to(array, (*array.shape[:-1], head_dim))

    history = [list(range(10, 10 + prefix_length)), []]
    warm = RotatingKVCache(max_size=window)
    if prefix_length:
        warm.update_and_fetch(
            mx.zeros((1, 1, prefix_length, head_dim)), values([history[0]])
        )
    batch = BatchRotatingKVCache.merge([warm, RotatingKVCache(max_size=window)])
    lengths = [3 if right_padded else 4, 4]
    if right_padded:
        batch.prepare(right_padding=[1, 0], lengths=lengths)

    suffixes = [[11, 12, 13, 0], [1, 2, 3, 4]]
    start = 0
    for size in parts:
        mask = batch.make_mask(size)
        keys, vals = batch.update_and_fetch(
            mx.zeros((2, 1, size, head_dim)),
            values([row[start : start + size] for row in suffixes]),
        )
        output = mx.fast.scaled_dot_product_attention(
            mx.zeros((2, 1, size, head_dim)), keys, vals, scale=1.0, mask=mask
        )
        actual = output[:, 0, :, 0].tolist()
        for row in range(2):
            for column in range(size):
                position = start + column
                if position < lengths[row]:
                    history[row].append(suffixes[row][position])
                    expected = history[row][-window:]
                    assert actual[row][column] == pytest.approx(
                        sum(expected) / len(expected), abs=1e-5
                    )
        start += size

    batch.finalize()
    assert batch._lengths is None
    assert batch.offset.tolist() == [len(row) for row in history]

    # After finalize(), normal one-token decoding must still rotate its mask.
    for step in range(window + 2):
        mask = batch.make_mask(1)
        new_values = [[100 + step], [200 + step]]
        keys, vals = batch.update_and_fetch(
            mx.zeros((2, 1, 1, head_dim)), values(new_values)
        )
        output = mx.fast.scaled_dot_product_attention(
            mx.zeros((2, 1, 1, head_dim)), keys, vals, scale=1.0, mask=mask
        )
        actual = output[:, 0, 0, 0].tolist()
        for row in range(2):
            history[row].extend(new_values[row])
            expected = history[row][-window:]
            assert actual[row] == pytest.approx(sum(expected) / len(expected), abs=1e-5)


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


@pytest.mark.parametrize("max_kv_size", [None, 1024])
def test_generation_forwards_max_kv_size(max_kv_size):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Mock(
                side_effect=self.__call__, layers=[object(), object()]
            )

        def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
            return InputEmbeddingsFeatures(
                inputs_embeds=mx.zeros((*input_ids.shape, 768))
            )

        def __call__(self, *args, **kwargs):
            return LanguageModelOutput(logits=mx.random.normal((1, 1, 32000)))

    cache = [
        SimpleNamespace(offset=0, state=(mx.zeros((1, 8, 100, 64)),) * 2)
        for _ in range(2)
    ]
    model = Model()
    with patch("mlx_vlm.models.cache.make_prompt_cache", return_value=cache) as make:
        list(
            generate_step(
                input_ids=mx.array([[1, 2, 3, 4, 5]]),
                model=model,
                pixel_values=mx.zeros((1, 3, 336, 336)),
                mask=mx.ones((1, 5)),
                kv_bits=4,
                max_kv_size=max_kv_size,
                max_tokens=1,
            )
        )
    make.assert_called_once_with(model.language_model, max_kv_size=max_kv_size)


@pytest.mark.parametrize("kv_bits", [None, 4])
def test_gemma2_attention_with_quantized_cache(kv_bits):
    from mlx_vlm.models.cache import KVCache, QuantizedKVCache
    from mlx_vlm.models.gemma2.config import ModelConfig
    from mlx_vlm.models.gemma2.language import Attention

    args = ModelConfig(
        model_type="gemma2",
        hidden_size=128,
        num_hidden_layers=1,
        intermediate_size=256,
        num_attention_heads=8,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=4,
    )
    attention = Attention(args)
    assert attention.repeats > 1

    cache = KVCache() if kv_bits is None else QuantizedKVCache(bits=kv_bits)
    x = mx.random.normal((1, 6, args.hidden_size))

    out = attention(x, mask=None, cache=cache)

    assert out.shape == (1, 6, args.hidden_size)
    assert not mx.any(mx.isnan(out))


@pytest.mark.parametrize("kv_bits", [None, 4])
def test_phi3small_block_sparse_attention_with_quantized_cache(kv_bits):
    from mlx_vlm.models.cache import KVCache, QuantizedKVCache
    from mlx_vlm.models.phi3small.config import ModelConfig
    from mlx_vlm.models.phi3small.language import Attention

    args = ModelConfig(
        model_type="phi3small",
        hidden_size=256,
        dense_attention_every_n_layers=2,
        ff_intermediate_size=512,
        gegelu_limit=20.0,
        num_hidden_layers=2,
        num_attention_heads=4,
        layer_norm_epsilon=1e-5,
        vocab_size=32,
        num_key_value_heads=2,
    )
    # layer 0 is a block-sparse layer, which is the hand-rolled path.
    attention = Attention(args, layer_idx=0)
    assert attention.block_sparse

    cache = KVCache() if kv_bits is None else QuantizedKVCache(bits=kv_bits)
    x = mx.random.normal((1, 6, args.hidden_size))

    out = attention(x, mask=None, cache=cache)

    assert out.shape == (1, 6, args.hidden_size)
    assert not mx.any(mx.isnan(out))


B, H, D = 2, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32
BITS = 8


def _rand_kv(batch, seq_len):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, H, seq_len, D))
    v = mx.random.normal((batch, H, seq_len, D))
    return k, v


def test_extend_handles_filtered_non_step_aligned_capacity():
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


def test_state_roundtrip():
    cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
    k, v = _rand_kv(B, 4)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys)

    state = cache.state
    assert len(state) == 4  # keys, values, offset, left_padding

    cache2 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
    cache2.state = state
    assert cache2._idx == 4


def test_empty_state():
    cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
    state = cache.state
    assert state[0] is None
    assert state[1] is None


def test_finalize_noop_without_prepare():
    cache = BatchQuantizedKVCache([1, 0], group_size=GROUP_SIZE, bits=BITS)
    k, v = _rand_kv(B, 4)
    cache.update_and_fetch(k, v)
    before = cache.left_padding.tolist()
    cache.finalize()
    assert cache.left_padding.tolist() == before


GROUP = 64


def test_str_passthrough():
    scores = mx.zeros((2, 8, 2, 4, 4))
    assert align_attention_mask_to_scores("causal", scores) == "causal"


def _quant_kv(B, n_kv, L, D, dtype=mx.float16):
    keys = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    values = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    return (
        mx.quantize(keys, group_size=GROUP, bits=BITS),
        mx.quantize(values, group_size=GROUP, bits=BITS),
    )


def _run_sdpa(B, n_q, n_kv, L, K_cache, mask, D=GROUP):
    """K_cache is total key length (offset + L); keys filled to K_cache."""
    queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    q_keys, q_values = _quant_kv(B, n_kv, K_cache, D)
    out = quantized_scaled_dot_product_attention(
        queries, q_keys, q_values, scale=D**-0.5, mask=mask, group_size=GROUP, bits=BITS
    )
    mx.eval(out)
    assert out.shape == (B, n_q, L, D)
    assert mx.isfinite(out).all()
    return out


HEAD_LAYOUTS = [
    (2, 16, 8),  # Qwen3-0.6B-like GQA (server repro family)
    (2, 32, 8),  # stronger GQA
    (2, 16, 2),  # wider repeat
    (2, 16, 1),  # MQA
    (2, 8, 8),  # MHA n_repeats=1
    (3, 16, 8),  # odd batch
    (4, 16, 8),  # larger batch
    (8, 16, 8),  # B == n_kv (latent mis-align case pre-fix)
    (1, 16, 8),  # single row control
]


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (3, 16, 8), (2, 16, 1)])
def test_left_and_right_padding_together(B, n_q, n_kv):
    L, offset = 16, 0
    left = mx.array([2, 0] + [0] * (B - 2))
    right = mx.array([0, 3] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, left_padding=left[:B], right_padding=right[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("window", [4, 8, 32])
@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (2, 8, 8)])
def test_sliding_window_causal_with_left_pad(window, B, n_q, n_kv):
    L, offset = 24, 40
    pads = mx.array([1, 0] if B == 2 else [1, 0] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, window_size=window, left_padding=pads[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (4, 32, 8)])
def test_additive_float_mask(B, n_q, n_kv):
    L = 12
    causal = create_causal_mask(L, left_padding=mx.array([i % 2 for i in range(B)]))
    mask = mx.where(
        causal, mx.array(0.0, dtype=mx.float16), mx.array(-1e4, dtype=mx.float16)
    )
    _run_sdpa(B, n_q, n_kv, L, L, mask)


def test_prepare_left_padding_on_empty_only():
    cache = BatchQuantizedKVCache([0, 0], group_size=GROUP, bits=BITS)
    cache.prepare(left_padding=[2, 1])
    assert cache.left_padding.tolist() == [2, 1]
    k = mx.random.normal((2, 2, 4, GROUP))
    v = mx.random.normal((2, 2, 4, GROUP))
    cache.update_and_fetch(k, v)
    with pytest.raises(ValueError, match="empty"):
        cache.prepare(left_padding=[1, 0])
