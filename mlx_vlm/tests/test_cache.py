"""Cache lifecycle, prefix reuse, persistence, quantization, and batched attention."""

from __future__ import annotations

import ast
import copy
import importlib
import logging
import os
import pkgutil
import shutil
import subprocess
import sys
import threading
from functools import partial
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models as models
import mlx_vlm.turboquant as tq
from mlx_vlm import apc as P
from mlx_vlm import apc_adapters as A
from mlx_vlm.apc import harvest_blocks_from_batch_cache as harvest
from mlx_vlm.apc import make_warm_batch_exact_cache_multi as warm_exact
from mlx_vlm.apc import make_warm_batch_kv_cache_multi as warm_blocks
from mlx_vlm.apc import snapshot_prompt_cache_row as snapshot_row
from mlx_vlm.apc_adapters import cache_memory_components
from mlx_vlm.apc_adapters import cache_memory_components as memory_components
from mlx_vlm.apc_adapters import clone_cache_entry
from mlx_vlm.generate import generate_step, maybe_quantize_kv_cache
from mlx_vlm.generate.ar import PromptProcessingBatch, _extend_cache, _make_cache
from mlx_vlm.models import cache as C
from mlx_vlm.models.base import (
    InputEmbeddingsFeatures,
    LanguageModelOutput,
    _turboquant_attention_applies,
    align_attention_mask_to_scores,
    quantized_scaled_dot_product_attention,
    scaled_dot_product_attention,
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
from mlx_vlm.models.minimax_m3_vl.language import (
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
)
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpAttention,
)
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache
from mlx_vlm.models.z1t.language import AFTConv, Z1TCache
from mlx_vlm.tests.test_models import DATA, build_config
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _SplitCodec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    resolve_kv_bits,
)
from mlx_vlm.vision_cache import VisionFeatureCache

# Cache lifecycle


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


@pytest.mark.parametrize(
    "factory", [BatchKVCache, BatchQuantizedKVCache, BatchQSAKVCache]
)
@pytest.mark.parametrize("trigger", ["prepare", "prefill", "ragged_commit"])
def test_qwen3_5_padding_mask_follows_in_place_updates(factory, trigger):
    from mlx_vlm.models.qwen3_5 import language as qwen3_5
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    def kv(steps):
        return mx.ones((2, 1, steps, 64))

    cache = factory([0, 0])
    decode = mx.zeros((2, 1, 16))
    if trigger == "prefill":
        cache.prepare(right_padding=[0, 3], lengths=[5, 2])
        cache.update_and_fetch(kv(4), kv(4))
    elif trigger == "ragged_commit":
        cache.update_and_fetch(kv(6), kv(6))

    assert qwen3_5._create_qwen3_5_attention_mask(decode, cache) is None
    if trigger == "prepare":
        cache.prepare(left_padding=[0, 3])
    elif trigger == "prefill":
        cache.update_and_fetch(kv(1), kv(1))
        cache.finalize()
    else:
        # QSA exposes the padding of its inner KV cache.
        transaction = start_speculative_cache([getattr(cache, "kv_cache", cache)], 4)
        cache.update_and_fetch(kv(4), kv(4))
        transaction.commit([4, 1])

    assert cache.left_padding.tolist() == [0, 3]
    assert qwen3_5._create_qwen3_5_attention_mask(decode, cache) == "left_padded_decode"
    assert cache._qwen3_5_decode_left_padding == [0, 3]
    assert qwen3_5._qwen3_5_left_padding_info(cache) == ((0, 3), 3)


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


def _rand_kv(batch, seq_len, heads=H):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, heads, seq_len, D))
    v = mx.random.normal((batch, heads, seq_len, D))
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


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_empty_quantized_cache_matches_quantize_layout(bits, head_dim, batched):
    if batched:
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=bits)
    else:
        cache = C.QuantizedKVCache(group_size=64, bits=bits)
    # The first update allocates the buffers, the second grows them.
    for steps in (3, 300):
        new = mx.random.normal((2 if batched else 1, 2, steps, head_dim))
        keys, _ = cache.update_and_fetch(new, new)
    expected = mx.quantize(new, group_size=64, bits=bits)
    assert [k.shape[-1] for k in keys] == [e.shape[-1] for e in expected]
    stored = mx.dequantize(*(k[..., 3:, :] for k in keys), group_size=64, bits=bits)
    assert mx.array_equal(stored, mx.dequantize(*expected, group_size=64, bits=bits))


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


class TestVisionFeatureCache:
    def test_lru_eviction(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("b.jpg", mx.ones((1, 10, 64)) * 2)
        cache.put("c.jpg", mx.ones((1, 10, 64)) * 3)  # evicts a.jpg
        assert cache.get("a.jpg") is None
        assert cache.get("b.jpg") is not None
        assert cache.get("c.jpg") is not None

    def test_multi_image_key(self):
        cache = VisionFeatureCache()
        features = mx.ones((1, 560, 1536))
        cache.put(["img1.jpg", "img2.jpg"], features)
        assert cache.get(["img1.jpg", "img2.jpg"]) is not None
        assert cache.get(["img2.jpg", "img1.jpg"]) is None  # order matters

    def test_contains(self):
        cache = VisionFeatureCache()
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        assert "a.jpg" in cache
        assert "b.jpg" not in cache

    def test_overwrite_existing_key(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("a.jpg", mx.ones((1, 10, 64)))
        cache.put("a.jpg", mx.ones((1, 10, 64)) * 5)
        assert len(cache) == 1
        result = cache.get("a.jpg")
        assert mx.array_equal(result, mx.ones((1, 10, 64)) * 5)


# Automatic prefix caching

parametrize = pytest.mark.parametrize


def forbid(monkeypatch, target, attr):
    monkeypatch.setattr(
        target, attr, lambda *a, **kw: pytest.fail(f"unexpected {attr}")
    )


def same_arrays(left, right):
    if isinstance(left, mx.array):
        assert (left.shape, left.dtype) == (right.shape, right.dtype)
        assert mx.array_equal(left, right).item()
    elif isinstance(left, (tuple, list, dict)):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            left, right = left.values(), right.values()
        for a, b in zip(left, right, strict=True):
            same_arrays(a, b)
    else:
        assert left == right


def same_cache(left, right):
    assert type(left) is type(right)
    if isinstance(left, (list, tuple)):
        for a, b in zip(left, right, strict=True):
            same_cache(a, b)
    else:
        same_arrays(left.state, right.state)
        same_arrays(left.meta_state, right.meta_state)
        if isinstance(left, C.CacheList):
            same_cache(left.caches, right.caches)


def kv(length=32, batch=1, heads=2, dim=32):
    pair = [mx.random.normal((batch, heads, length, dim)) for _ in range(2)]
    mx.eval(pair)
    return pair


def filled(cache, length=32, batch=1, heads=2, dim=32):
    cache.update_and_fetch(*kv(length, batch, heads, dim))
    mx.eval(cache.state)
    return cache


def allocated(length, value=1):
    cache = C.KVCache()
    cache.step = 1
    cache.keys = mx.full((1, 1, length, 4), value, dtype=mx.float32)
    cache.values = cache.keys + 1
    cache.offset = length
    mx.eval(cache.state)
    return cache


def batch_cache(kind, padding=(0,), **kwargs):
    factory, defaults = {
        "dense": (C.BatchKVCache, {}),
        "rotating": (C.BatchRotatingKVCache, {"max_size": 64}),
        "uniform": (C.BatchQuantizedKVCache, {"group_size": 32, "bits": 8}),
        "turbo": (BatchTurboQuantKVCache, {"bits": 4.0}),
    }[kind]
    return factory(left_padding=list(padding), **(defaults | kwargs))


def clone(cache):
    targets = []
    result = A.clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=targets)
    mx.eval(targets)
    return result


def cache_model(caches):
    return NS(make_cache=lambda: caches)


def coordinate(manager, caches):
    return manager.coordinator(cache_model(caches))


@pytest.fixture
def managers(tmp_path, monkeypatch):
    """Own writers and readers, including reopened namespaces and memory budgets."""
    owned = []

    def make(
        tier="memory", *, blocks=8, block=16, namespace="unit", budget=None, **settings
    ):
        disk = None
        if tier != "memory":
            disk = P.DiskBlockStore(tmp_path, namespace=namespace)
        manager = P.APCManager(num_blocks=blocks, block_size=block, disk=disk)
        if tier == "disk-only":
            manager._exact_cache_max = 0
        if budget is not None:
            manager.memory_max_bytes = budget
            manager.memory_reserve_bytes = 0
            monkeypatch.setattr(manager, "_memory_headroom", lambda: 1 << 40)
        for key, value in settings.items():
            setattr(manager, key, value)
        owned.append(manager)
        return manager

    yield make
    for manager in reversed(owned):
        manager.close()


@pytest.fixture
def memory_manager(managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "16")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    return lambda budget=4096, disk=False: managers(
        "disk" if disk else "memory", budget=budget, blocks=16
    )


@pytest.fixture
def prefix_manager(managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    return managers


def disk_roundtrip(managers, tokens, caches, **kwargs):
    writer = managers("disk", **kwargs)
    assert writer.store_exact_cache(tokens, caches)
    writer.disk.flush()
    writer.close()
    reader = managers("disk", **kwargs)
    restored, count = reader.lookup_exact_cache(tokens + [999])
    assert count == len(tokens) and restored is not None
    assert reader.stats.disk_hits == 1
    return restored, reader


def store_blocks(manager, tokens, layers=2, dim=4, extra_hash=0):
    pairs = [kv(len(tokens), heads=1, dim=dim) for _ in range(layers)]
    return manager.store_kv_blocks(
        tokens, *map(list, zip(*pairs)), extra_hash=extra_hash
    )


def apc_config(name, *, text_only=False):
    profile = DATA["apc"][name]
    cases = {c["id"]: c for c in DATA["cases"]}
    case = cases[profile["case"]] if "case" in profile else profile

    fields = copy.deepcopy(case["config"])

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(fields, profile["config"])
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    if text_only:
        return module, build_config(module, fields["text_config"], "TextConfig")
    return module, build_config(module, fields)


def language_model(name):
    module, config = apc_config(name)
    if name == "qwen3_5":
        return module.LanguageModel(config.text_config, config)
    return module.Model(config).language_model


def test_hashes_and_dependencies():
    tokens = tuple(range(16))
    hash_tokens = P._hash_tokens
    assert hash_tokens(0, tokens, 0) == hash_tokens(0, tokens, 0)
    variants = [(0, 0), (0, 42), (7, 0), (8, 0)]
    assert len({hash_tokens(seed, tokens, extra) for seed, extra in variants}) == 4
    image_hash, tenant_hash = P.hash_image_payload, P.tenant_scoped_hash
    assert image_hash(pixel_values=mx.zeros((1, 3, 8, 8))) != image_hash(
        pixel_values=mx.ones((1, 3, 8, 8))
    )
    refs = ["a.png", "b.png"]
    assert image_hash(image_ref=refs) == image_hash(image_ref=refs)
    assert image_hash(None, None) == 0
    assert P._hash_payload(None) is P._hash_payload([]) is None
    assert P._hash_payload(refs) == P._hash_payload(refs)
    assert P._hash_payload("x") == image_hash(image_ref="x")
    image = image_hash(image_ref="cat.jpg")
    assert tenant_hash(None, image) == image
    assert tenant_hash("a", image) == tenant_hash("a", image)
    variants = [("a", image), ("b", image), ("a", 42)]
    assert len({tenant_hash(t, i) for t, i in variants}) == 3
    code = "from mlx_vlm.apc import tenant_scoped_hash; print(tenant_scoped_hash('a', 123456789))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1]
    base = P.semantic_extra_hash(image_hash=5)
    good = NS(apc_key_dependencies=lambda: ["adapter-x"])
    broken = NS(apc_key_dependencies=lambda: (_ for _ in ()).throw(ValueError))
    assert P.semantic_extra_hash(image_hash=5, model=good) != base
    for model in (NS(), NS(apc_key_dependencies=5), broken):
        assert P.semantic_extra_hash(image_hash=5, model=model) == base
    assert P.model_key_dependencies(None, None) == ()


def test_blocks_and_statistics(managers):
    manager = managers(blocks=16)
    tokens = list(range(53))
    assert manager.lookup_prefix(tokens) == ([], 0)
    stored = store_blocks(manager, tokens)
    assert len(stored) == 3
    manager.release(stored)
    for _ in range(2):
        matched, count = manager.lookup_prefix(tokens)
        assert len(matched) == 3 and count == 48
        warm = P.make_warm_kv_cache(matched, min_capacity_tokens=65)
        assert len(warm) == 2
        assert all(
            c.offset == 48 and c.keys.shape[:2] == (1, 1) and c.keys.shape[2] >= 65
            for c in warm
        )
        manager.release(matched)
        assert manager.stats_snapshot()["lookups_hit"] == 1
        manager.reset_stats()
        assert manager.stats_snapshot()["lookups_hit"] == 0
    manager.clear()
    assert manager.stats_snapshot()["pool_used"] == 0
    assert manager.lookup_prefix(tokens) == ([], 0)


def test_layer_major_threshold(managers, monkeypatch):
    monkeypatch.setenv("APC_LAYER_MAJOR_MEMORY_MIN_TOKENS", "1")
    manager = managers(blocks=16)
    tokens = list(range(64))
    sources = [allocated(64, value) for value in (1, 3)]
    keys, values = [c.keys for c in sources], [c.values for c in sources]
    assert manager.store_kv_blocks(tokens, keys, values) == []
    assert manager.lookup_prefix(tokens)[1] == 0
    warm, count = manager.lookup_exact_cache(tokens + [999])
    assert count == 48 and len(warm) == 2
    for restored, source in zip(warm, sources):
        assert restored.offset == 48 and restored.keys.shape[2] >= 65
        same_arrays(restored.keys[..., :48, :], source.keys[..., :48, :])
        same_arrays(restored.values[..., :48, :], source.values[..., :48, :])


def test_disk_block_lifecycle(managers, monkeypatch):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "1")
    manager = managers("disk", blocks=1)
    first, second = list(range(48)), list(range(100, 148))
    manager.release(store_blocks(manager, first))
    manager.disk.flush()
    assert any(manager.disk.dir.glob(f"*{manager.disk.SUFFIX}"))
    shutil.rmtree(manager.disk.dir)
    assert not manager.disk.dir.exists()
    manager.release(store_blocks(manager, second))
    manager.disk.flush()
    size = manager.disk.disk_bytes
    assert size > 0 and manager.disk.dir.exists()
    manager.close()
    reader = managers("disk")
    warm, count = reader.lookup_prefix_disk_cache(second)
    assert count == 48 and all(c.offset == 48 for c in warm)
    assert reader.stats_snapshot()["pool_used"] == 0
    reader.disk.max_bytes = int(size * 0.75)
    assert reader.disk._maybe_evict() > 0
    warm, count = reader.lookup_prefix_disk_cache(second)
    assert warm is not None and 0 < count < 48


def test_disk_policy_and_metadata(managers, monkeypatch):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "3")
    manager = managers("disk")
    tokens = list(range(48))
    manager.release(store_blocks(manager, tokens))
    manager.disk.flush()
    assert manager.lookup_prefix_disk_cache(tokens) == (None, 0)
    warm, count = manager.lookup_prefix_disk_cache(
        tokens, allow_memory_overlap=True, max_prefix_tokens=32, min_prefix_tokens=16
    )
    assert warm is not None and count == 32
    assert manager.lookup_prefix_disk_cache(
        tokens, allow_memory_overlap=True, max_prefix_tokens=32, min_prefix_tokens=32
    ) == (None, 0)
    manager._disk_min_free_ram_bytes = 2
    monkeypatch.setattr(P, "_free_ram_bytes", lambda: 1)
    lookup = manager.lookup_prefix_disk_cache
    assert lookup(tokens, allow_memory_overlap=True) == (None, 0)
    monkeypatch.undo()
    writer = managers("disk", namespace="metadata")
    tokens = list(range(16))
    writer.release(store_blocks(writer, tokens, extra_hash=1))
    writer.disk.flush()
    writer.close()
    reader = managers("disk", namespace="metadata")
    assert reader.lookup_prefix_disk_cache(tokens, extra_hash=2) == (None, 0)
    wrong, real = [P._hash_tokens(0, tuple(tokens), extra) for extra in (2, 1)]
    reader.disk._index[wrong] = reader.disk._index[real]
    assert reader.lookup_prefix_disk_cache(tokens, extra_hash=2) == (None, 0)


def test_exact_promotion_and_priority(managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    tokens = [list(range(20)), list(range(100, 120))]
    writer = managers("disk")
    for value, ids in enumerate(tokens, 1):
        assert writer.store_exact_cache(ids, [allocated(20, value)])
    writer.disk.flush()
    writer.close()
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1")
    reader = managers("disk")
    for index, hits in [(0, 1), (0, 1), (1, 2), (0, 3)]:
        warm, count = reader.lookup_exact_cache(tokens[index] + [999])
        assert warm is not None and count == 20 and reader.stats.disk_hits == hits
    source = allocated(20, 99)
    assert reader.store_exact_cache(tokens[0], [source])
    warm, count = reader.lookup_exact_cache(tokens[0] + [999])
    assert count == 20 and reader.stats.disk_hits == 3
    assert reader.stats.exact_stores == 1
    same_arrays(warm[0].keys[..., :20, :], source.keys)


def test_self_check_and_plans(managers, monkeypatch, caplog):
    monkeypatch.delenv("APC_TRACE", raising=False)
    layouts = [
        ([C.KVCache(), C.KVCache()], "block"),
        ([C.ArraysCache(2), C.KVCache()], "exact"),
        ([C.RotatingKVCache(8), C.KVCache()], "exact"),
        ([object()], None),
    ]
    assert P.model_apc_mode(object()) == "block"
    assert A.apc_mode([]) is None
    for caches, mode in layouts:
        model = cache_model(caches)
        assert P.model_apc_mode(model) == A.apc_mode(caches) == mode
        if mode is None:
            continue
        plan = A.build_prefix_cache_plan(model)
        assert plan.restorable and len(plan.components) == 2
        assert plan.is_hybrid is (mode == "exact")
        assert plan.strategy == ("checkpoint" if mode == "exact" else "block")
        assert len(plan.groups) == (2 if mode == "exact" else 1)
        if isinstance(caches[0], C.ArraysCache):
            assert plan.capabilities == [A.Capability.CHECKPOINT, A.Capability.PAGEABLE]
        if mode == "block":
            assert plan.groups[0].layer_indices == (0, 1)
        assert "PrefixCachePlan" in plan.describe()
    plan = A.build_prefix_cache_plan(NS(language_model=NS(layers=[object()] * 3)))
    assert plan.restorable and plan.strategy == "block"
    assert len(plan.components) == 3 and len(plan.groups) == 1
    assert (
        P.classify_layer_for_apc(filled(C.QuantizedKVCache(), 8, dim=64)).status == "ok"
    )
    rejected = P.classify_layer_for_apc(object())
    assert rejected.status == "unsupported" and rejected.reason
    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        result = P.self_check_model_apc(
            cache_model([batch_cache(k) for k in ("rotating", "uniform", "dense")]),
            kv_bits=8.0,
        )
    assert result.ok and result.apc_mode == "exact"
    assert any("APC self-check ok" in r.message for r in caplog.records)
    assert not P.self_check_model_apc(object()).ok
    broken = P.self_check_model_apc(
        NS(make_cache=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    assert not broken.ok and broken.notes
    monkeypatch.setenv("APC_TRACE", "1")
    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        assert not managers().store_exact_cache(
            list(range(16)), [NS(keys="bad", values="bad")]
        )
    assert any("APC_TRACE reject" in r.message for r in caplog.records)
    assert any("unclonable" in r.message for r in caplog.records)


@parametrize("kind", ["dense", "rotating", "uniform", "turbo"])
def test_batch_protocol(kind):
    cache = batch_cache(kind, [0, 0])
    if kind in ("rotating", "turbo"):
        assert cache.empty() and cache.batch_size == 2 and not cache.is_single_row()
        single = filled(batch_cache(kind), 8)
        assert not single.empty() and single.batch_size == 1 and single.is_single_row()
    if kind in ("uniform", "turbo"):
        expected = C.QuantizedKVCache if kind == "uniform" else TurboQuantKVCache
        row = cache.extract(0)
        assert isinstance(row, expected) and (row.keys is None or row.offset == 0)
        assert row.dequantize_for_apc() == (None, None)
    if kind == "turbo":
        assert cache.dequantize_for_apc() == (None, None)
        keys, values = kv(24, batch=2)
        cache.update_and_fetch(keys, values)
        row = cache.extract(1)
        assert isinstance(row, TurboQuantKVCache) and row.offset == 24
        k, v = row.dequantize_for_apc()
        assert k.shape == (1, 2, 24, 32)
        assert mx.max(mx.abs(k - keys[1:2])).item() < 2
    else:
        cache.prepare(right_padding=[2, 0], lengths=[4, 6])
        cache.update_and_fetch(*kv(3, batch=2))
        cache.filter(mx.array([1, 0], dtype=mx.int32))
        cache.filter(mx.array([1], dtype=mx.int32))
        if kind == "rotating":
            assert cache._lengths.tolist() == [4]
            assert all(x.shape[0] == 1 for x in cache.update_and_fetch(*kv(2)))
        else:
            assert cache._right_padding.tolist() == [2]
        cache.finalize()
        assert cache.offset.tolist() == [4 if kind == "rotating" else 1]
        assert cache.left_padding.tolist() == [1 if kind == "rotating" else 2]
    keys, values = P.layer_kv_for_apc(cache, batch_idx=0)
    assert keys.shape[0] == values.shape[0] == 1
    assert not isinstance(keys, tuple)
    single = filled(batch_cache("rotating"), 16)
    row = P.extract_prompt_cache_from_batch([single], 0)
    assert P._clone_prompt_cache_for_apc(row) is not None


def test_float_extraction_and_harvesting(managers):
    for cache in (C.KVCache(), C.QuantizedKVCache(32, 8)):
        keys, values = P.layer_kv_for_apc(filled(cache, 12))
        assert keys.shape == values.shape == (1, 2, 12, 32)
    bogus = NS(keys=(1, 2, 3), values=(4, 5, 6), offset=3)
    assert P.layer_kv_for_apc(bogus) == (None, None)
    source, target = managers(), managers()
    full, short = list(range(32)), list(range(100, 116))
    blocks = [store_blocks(source, ids) for ids in (full, short)]
    cache, _ = warm_blocks(
        [
            {"matched_blocks": b, "prefix_len": len(ids)}
            for b, ids in zip(blocks, (full, short))
        ],
        num_layers=2,
    )
    harvested = harvest(target, cache, batch_idx=1, full_token_ids=short)
    assert len(harvested) == 1
    same_arrays(harvested[0].keys[0], blocks[1][0].keys[0])
    matched, count = target.lookup_prefix(short)
    assert count == 16
    target.release(matched + harvested)
    source.release(blocks[0] + blocks[1])
    quantized = filled(batch_cache("uniform", [3]), 35)
    harvested = harvest(target, [quantized] * 2, full, batch_idx=0)
    assert len(harvested) == 2
    assert all(k.shape[2] == 16 for block in harvested for k in block.keys)
    target.release(harvested)
    assert harvest(target, [batch_cache("uniform")], list(range(16)), batch_idx=0) == []


def test_pooling_merge():
    rows = []
    for length in (6, 0):
        rotating, pooling = C.RotatingKVCache(16), C.PoolingCache(4)
        if length:
            keys = mx.arange(length * 3, dtype=mx.float32).reshape(1, 1, length, 3)
            rotating.update_and_fetch(keys, keys + 1)
            ready, _, _ = pooling.accumulate_windows(
                keys.reshape(1, length, 3), mx.ones((1, length, 2)), offset=0
            )
            pooling.update_and_fetch(mx.ones((1, ready.shape[1] // 4, 3)))
        rows.append(snapshot_row([C.CacheList(rotating, pooling)], 0))
    merged, count = warm_exact(rows, [6, 0])
    assert count == 6
    rotating, pooling = merged[0].caches
    assert isinstance(rotating, C.BatchRotatingKVCache)
    assert rotating.offset.tolist() == [6, 0]
    direct = A.merge_cache_entries([row[0].caches[1] for row in rows], [6, 0])
    for cache in (pooling, direct):
        assert isinstance(cache, C.BatchPoolingCache) and cache.ratio == 4
        assert (cache.remainder, cache._pool_lengths, cache._processed) == (
            [2, 0],
            [1, 0],
            [6, 0],
        )
        assert cache.extract(0).remainder == 2
        assert cache.extract(0).pooled.shape == (1, 1, 3)
        assert cache.extract(1).empty()
    empty = C.BatchPoolingCache.merge(
        [C.PoolingCache(4), C.PoolingCache(4)], prefix_lens=[0, 0]
    )
    assert empty.remainder == empty._pool_lengths == empty._processed == [0, 0]


@parametrize("slot", [None, 0, 1])
def test_optional_array_slots(slot):
    rows = [C.ArraysCache(2) for _ in range(3)]
    if slot is not None:
        rows[0][slot], rows[2][slot] = mx.ones((1, 4)), mx.full((1, 4), 2.0)
    merged = A.merge_cache_entries(rows, [4, 0, 4])
    assert merged.empty() is (slot is None)
    if slot is None:
        assert merged.cache == [None, None]
        assert merged.left_padding.tolist() == [0, 0, 0]
    else:
        assert merged[1 - slot] is None
        assert merged[slot].tolist() == [[1] * 4, [0] * 4, [2] * 4]


@parametrize(
    "kind,length",
    [("buffered", n) for n in (0, 4, 48)]
    + [("ring", 6), ("indexed", 5), ("chunked", 6)],
)
def test_clone_continuation(kind, length):
    cache = {
        "buffered": lambda: C.BufferedRotatingKVCache(8, buffer_size=3),
        "ring": lambda: RingSlidingKVCache(4),
        "indexed": MiniMaxM3KVCache,
        "chunked": lambda: C.ChunkedKVCache(4),
    }[kind]()
    if kind == "buffered":
        for token in range(length):
            keys = mx.full((1, 1, 1, 4), token, dtype=mx.float32)
            cache.update_and_fetch(keys, keys + 1)
    else:
        filled(cache, length, heads=2, dim=8)
    if kind == "ring":
        assert A.apc_exact_eligible(cache) and not A.apc_block_eligible(cache)
        for _ in range(7):
            filled(cache, 1, heads=2, dim=8)
    if kind == "indexed":
        cache.update_index_and_fetch(kv(5, dim=8)[0])
        assert A.apc_exact_eligible(cache)
        assert A.merge_cache_entries([cache, clone(cache)], [5, 5]) is not None
    if kind == "chunked":
        cache.maybe_trim_front()
        restored = C.ChunkedKVCache(4)
        A.CheckpointAdapter().restore(restored, A.CheckpointAdapter().capture(cache, 6))
        assert (restored.offset, restored.start_position) == (6, 2)
    else:
        restored = clone(cache)
        assert type(restored) is type(cache)
        assert restored.meta_state == cache.meta_state
    for size in ((3, 12, 1) if kind == "buffered" else (1,)):
        if kind == "buffered":
            same_arrays(
                restored.make_mask(2, return_array=True),
                cache.make_mask(2, return_array=True),
            )
        keys, values = kv(
            size,
            heads=1 if kind == "buffered" else 2,
            dim=4 if kind == "buffered" else 8,
        )
        same_arrays(
            restored.update_and_fetch(keys, values),
            cache.update_and_fetch(keys, values),
        )
        if kind == "buffered":
            restored.trim(1)
            cache.trim(1)
            assert restored.meta_state == cache.meta_state
    if kind == "ring":
        assert A.merge_cache_entries([cache], [cache.offset]) is None


def sample(name, length=0):
    constructors = {
        "ArraysCache": lambda: C.ArraysCache(2),
        "CacheList": lambda: C.CacheList(
            sample("KVCache", length), sample("ArraysCache", length)
        ),
        "ChunkedKVCache": lambda: C.ChunkedKVCache(8),
        "PoolingCache": lambda: C.PoolingCache(2),
        "RingSlidingKVCache": lambda: RingSlidingKVCache(max(16, length)),
        "RotatingKVCache": lambda: C.RotatingKVCache(max(16, length * 2)),
        "StaticPrefixKVCache": lambda: C.StaticPrefixKVCache(max(16, length)),
    }
    cache = constructors.get(name, getattr(C, name, globals().get(name)))()
    if not length or name == "CacheList":
        return cache
    keys = mx.ones((1, 1, length, 4))
    if name == "ArraysCache":
        cache.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
    elif name == "PoolingCache":
        cache.pooled = mx.ones((1, length // 2, 4))
        cache.buf_kv, cache.buf_gate = mx.ones((1, 2, 4)), mx.ones((1, 2, 1))
        cache.remainder = 1
    elif name == "Z1TCache":
        cache.offset = length
        cache.cum_eKV, cache.cum_eK = mx.ones((1, 4)), mx.ones((1, 4)) * 2
        cache.win_eKV, cache.win_eK = mx.ones((1, 3, 4)) * 3, mx.ones((1, 3, 4)) * 4
    elif name in ("SimpleKVCache", "MiniMaxM3KVCache"):
        cache.update_and_fetch(keys, keys * 2)
        if name == "MiniMaxM3KVCache":
            cache.update_index_and_fetch(keys)
    else:
        cache.keys, cache.values, cache.offset = keys, keys * 2, length
        if name == "RotatingKVCache":
            cache._idx = length
        if name == "QSAKVCache":
            cache.index_keys = mx.ones((1, length, 4)) * 3
            cache.index_position_ids = mx.arange(length, dtype=mx.int64)[None]
    return cache


def cache_names(tree):
    names = {
        getattr(n.func, "id", getattr(n.func, "attr", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    return {name for name in names if name.endswith("Cache") or name == "CacheList"}


def discover_contracts():
    """Distinct local factories cover wrappers sharing imported language backbones."""
    root = Path(models.__file__).resolve().parent
    local, generative = {}, 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "make_cache"
            ):
                local.setdefault(path.relative_to(root).parts[0], set()).update(
                    cache_names(node)
                )

    for info in pkgutil.iter_modules(models.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            cls = getattr(
                importlib.import_module("mlx_vlm.models." + info.name), "Model", None
            )
        except ModuleNotFoundError:
            continue
        generative += callable(getattr(cls, "get_input_embeddings", None))
    assert generative >= 120 and len(local) >= 70
    return sorted({("KVCache",)} | {tuple(sorted(v)) for v in local.values() if v})


@parametrize("names", discover_contracts(), ids=lambda names: "+".join(names))
def test_model_cache_contract(names, managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    for name in names:
        cache = sample(name)
        assert A.build_prefix_cache_plan_from_caches([cache]).restorable
        restored = clone(cache)
        assert restored is not None
        assert all(not p.fallback for p in memory_components([restored], 0))
    caches = [sample(name, 16) for name in names]
    plan = A.build_prefix_cache_plan_from_caches(caches)
    assert plan.restorable, plan.describe()
    manager = managers(block=8)
    runner = coordinate(manager, caches)
    assert runner.strategy == plan.strategy
    tokens = list(range(16))
    if plan.strategy == "block":
        keys, values = [c.keys for c in caches], [c.values for c in caches]
        manager.release(manager.store_kv_blocks(tokens, keys, values))
    else:
        assert manager.store_exact_cache(tokens, caches)
    hit = runner.lookup(
        tokens + [999],
        extra_hash=0,
        safe_lookup_min=0,
        suffix_is_text_only=lambda _: True,
        prefix_has_media=lambda _: False,
    )
    assert hit is not None and hit["prefix_len"] == 16
    counter = "lookups_hit" if plan.strategy == "block" else "exact_hits"
    assert manager.stats_snapshot()[counter] == 1
    if plan.strategy != "block":
        assert hit["warm_cache"] is not None
    assert manager.stats_snapshot()["restored_tokens"] == 0
    assert runner.materialize_single(hit, min_capacity_tokens=17) is not None
    assert manager.stats_snapshot()["restored_tokens"] == 16
    manager.reset_stats()
    merged, _ = runner.merge_rows([hit], [16])
    assert manager.stats_snapshot()["restored_tokens"] == (
        16 if merged is not None else 0
    )
    runner.release_hit(hit)


@parametrize("kind", ["qsa", "deepseek", "composite", "ring-indexed"])
def test_disk_custom_state(kind, managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1" if kind == "qsa" else "0")
    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")
    if kind == "qsa":
        arrays, qsa = C.ArraysCache(1), sample("QSAKVCache", 40)
        arrays[0] = mx.arange(6, dtype=mx.int64).reshape(1, 2, 3)
        qsa.index_position_ids = mx.arange(120, dtype=mx.int64).reshape(3, 1, 40)
        qsa.index_block_keys, qsa.index_block_ratio = mx.ones((1, 1, 10, 4)), 4
        caches, length = [arrays, qsa], 40
    elif kind == "deepseek":
        rotating = C.RotatingKVCache(8, keep=0)
        rotating.keys, rotating.values = mx.ones((1, 1, 8, 4), mx.bfloat16), mx.zeros(
            (1, 1, 8, 0), mx.bfloat16
        )
        rotating.offset, rotating._idx = 40, 3
        pools = [C.PoolingCache(4), C.PoolingCache(4)]
        pools[0].pooled, pools[1].pooled = mx.ones((1, 6, 4), mx.bfloat16), mx.ones(
            (1, 6, 2)
        )
        caches, length = [C.CacheList(rotating, *pools)], 40
    elif kind == "composite":
        caches, length = [
            (sample("SimpleKVCache", 12), C.SimpleKVCache()),
            C.CacheList(sample("PoolingCache", 10)),
        ], 12
    else:
        ring = RingSlidingKVCache(4)
        ring.keys, ring.values = mx.ones((1, 2, 8, 4)), mx.ones((1, 2, 8, 4)) * 2
        ring.prefill_length, ring.offset, ring._ring_pos = 4, 11, 3
        caches, length = [ring, sample("MiniMaxM3KVCache", 10)], 10
    restored, reader = disk_roundtrip(managers, list(range(length)), caches, block=4)
    same_cache(restored, caches)
    if kind == "qsa":
        assert restored[1].keys.shape[2] >= 41 and restored[1].offset == 40
        memory, count = reader.lookup_exact_cache(list(range(40)) + [998])
        assert count == 40 and memory[1].keys.shape[2] >= 41
        assert reader.stats.disk_hits == 1
        batch, count = warm_exact([restored], [40])
        assert count == 40 and isinstance(batch[1], BatchQSAKVCache)
        same_cache(batch[1].extract(0), qsa)
        path = next(iter(reader.disk._exact_index.values()))
        assert reader.disk._open_shard_header(path)[1]["c1_kind"] == "checkpoint"


@parametrize("family", ["builtin", "model", "qsa", "pooling", "minimax"])
def test_memory_growth_profiles(family):
    factories = {
        "builtin": [
            C.ConcatenateKVCache,
            C.SimpleKVCache,
            partial(C.ChunkedKVCache, 65),
            partial(C.BufferedRotatingKVCache, 65, buffer_size=300),
            partial(C.BufferedRotatingKVCache, 65, keep=1),
            partial(C.StaticPrefixKVCache, 513),
        ],
        "model": [HyV4KVCache, partial(RingSlidingKVCache, 16)],
        "qsa": [
            QSAKVCache,
            partial(QSAQuantizedKVCache, 32, 4),
            partial(BatchQSAKVCache, [0, 0]),
            partial(BatchQSAKVCache, [0, 3]),
        ],
        "pooling": [partial(C.PoolingCache, r) for r in (4, 64)]
        + [partial(C.BatchPoolingCache, r, [0, 3, 7]) for r in (4, 64)],
        "minimax": [MiniMaxM3KVCache, partial(MiniMaxM3BatchKVCache, [0, 3])],
    }[family]
    chunks = {"builtin": [37, 256], "model": [37, 256, 1024], "pooling": [37]}.get(
        family, [257]
    )
    seeds = {"builtin": [1], "qsa": [1, 16], "pooling": [1, 65]}.get(family, [16])
    for (index, factory), chunk, seed in product(enumerate(factories), chunks, seeds):
        mrope = family == "qsa" and index in (1, 2)
        cache = factory()
        batch = len(cache.left_padding) if hasattr(cache, "left_padding") else 1
        ratio = cache.ratio if family == "pooling" else 0

        def advance(start, length):
            if family == "pooling":
                ready, _, _ = cache.accumulate_windows(
                    mx.ones((batch, length, 8), mx.float16),
                    mx.ones((batch, length, 4)),
                    0,
                )
                cache.update_and_fetch(
                    mx.ones((batch, ready.shape[1] // ratio, 4), mx.float16)
                )
                return
            if family == "qsa":
                positions = mx.broadcast_to(
                    mx.arange(start, start + length), (batch, length)
                )
                if mrope:
                    positions = mx.broadcast_to(positions, (3, batch, length))
                indexer.select_from_projected(
                    mx.ones((batch, length, 24)), cache, positions
                )
            kd, vd = (32, 64) if family == "qsa" else (4, 8)
            dtype = mx.float16 if family in ("qsa", "minimax") else mx.float32
            if isinstance(cache, C.ChunkedKVCache):
                cache.maybe_trim_front()
            cache.update_and_fetch(
                mx.ones((batch, 1, length, kd), dtype),
                mx.ones((batch, 1, length, vd), dtype),
            )
            if family == "minimax":
                cache.update_index_and_fetch(mx.ones((batch, 1, length, 8)))

        if family == "qsa":
            indexer = Qwen4ExpAttention(
                apc_config("qwen4_exp", text_only=True)[1]
            ).indexer
        empty = memory_components([cache], 0)[0]
        assert not empty.fallback and empty.footprint(6000, chunk) == 0
        advance(0, seed)
        if family in ("qsa", "model") or (family == "minimax" and batch == 1):
            cache = clone(cache)
        profile = memory_components([cache], seed, batch_size=batch)[0]
        assert not profile.fallback
        if family == "pooling":
            assert profile.fixed_bytes == ratio * 32
        else:
            assert profile.source_bytes * batch == cache.nbytes
        peak = cache.nbytes
        if family == "builtin":
            assert peak <= profile.footprint(1, chunk) < peak + 24576
        for start in range(seed, 6000, chunk):
            advance(start, min(chunk, 6000 - start))
            peak = max(peak, cache.nbytes)
        estimate = batch * profile.footprint(6000, chunk)
        measured = peak if family == "builtin" else cache.nbytes
        upper = (
            measured + 24576
            if family == "builtin"
            else measured * {"qsa": 1.3, "minimax": 1.1}.get(family, 2)
        )
        assert measured <= estimate, (family, chunk, seed, batch)
        assert estimate <= upper if family in ("pooling", "model") else estimate < upper
        if family == "builtin":
            assert profile.footprint(0, chunk) == 0


@parametrize("read_only", [False, True])
def test_static_and_fixed_profiles(read_only):
    prefix = filled(C.StaticPrefixKVCache(513), 16, heads=1, dim=4)
    source = C.StaticPrefixKVCache.from_prefix(prefix) if read_only else prefix
    cache = C.StaticPrefixKVCache.from_state(source.state, source.meta_state)
    profile = memory_components([cache], 16)[0]
    assert not profile.fallback and cache.read_only == read_only
    if read_only:
        assert profile.footprint(1) == profile.footprint(6000) == cache.nbytes
    filled(cache, 1, heads=1, dim=4)
    assert cache.offset == (16 if read_only else 17)
    assert not C.StaticPrefixKVCache.from_state(
        source.state, source.meta_state[:3]
    ).read_only
    layer, cache = AFTConv(apc_config("z1t")[1]), Z1TCache()
    layer(mx.ones((1, 1, 8)), cache)
    cache = clone(cache)
    profile = memory_components([cache], 1)[0]
    assert not profile.fallback
    layer(mx.ones((1, 5999, 8)), cache)
    assert profile.footprint(1) == profile.footprint(6000) == P._cache_nbytes(cache)


@parametrize("factory", [C.KVCache, C.QuantizedKVCache, lambda: C.RotatingKVCache(512)])
def test_unused_capacity_reservation(factory, memory_manager):
    cache = factory()
    cache.step = 1
    filled(cache, 256, heads=1, dim=64)
    cache.trim(240)
    cache.step = 256
    size = P._cache_nbytes(cache)
    manager = memory_manager(budget=1 << 20)
    assert manager.store_exact_cache(list(range(16)), [cache])
    coordinate(manager, [cache]).prepare_prefill([2000] * 3)
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 6 * 2048 * (size // 256)
    assert P._cache_nbytes(cache) == size == 16 * P._cache_nbytes(cache.state)


@parametrize(
    "operation",
    ["disk_expand", "padded", "batch", "exact_load", "block_load", "memory_expand"],
)
def test_admission_before_allocation(operation, memory_manager, monkeypatch):
    disk = operation in ("disk_expand", "batch", "exact_load", "block_load")
    budgets = dict(padded=1024, exact_load=0, block_load=0)
    budget = budgets.get(operation, 4096)
    manager = memory_manager(budget=budget, disk=disk)
    tokens, source = list(range(16)), allocated(16)
    if operation in ("disk_expand", "memory_expand"):
        assert manager.store_exact_cache(tokens, [source])
        if disk:
            manager.disk.flush()
            manager = memory_manager(disk=True)
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 4096)
        attr = "prefix_cache_reserve" if disk else "_clone_prompt_cache_for_apc"
        forbid(monkeypatch, C.KVCache if disk else P, attr)
        request = tokens + [99] * (6000 if disk else 984)
        assert manager.lookup_exact_cache(request) == (None, 0)
    elif operation.endswith("load"):
        source, tokens = allocated(32), list(range(32))
        if operation == "exact_load":
            manager.store_exact_cache(tokens, [source])
            loader, lookup = "load_exact_cache", manager.lookup_exact_cache
        else:
            manager.store_kv_blocks(tokens, [source.keys], [source.values])
            loader, lookup = "load_layer_major_prefix", manager.lookup_prefix_disk_cache
        manager.disk.flush()
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
        forbid(monkeypatch, manager.disk, loader)
        assert lookup(tokens + [99]) == (None, 0)
    elif operation == "padded":
        source = allocated(256)
        source.offset = 16
        forbid(monkeypatch, P, "_clone_prompt_cache_for_apc")
        assert not manager.store_exact_cache(tokens, [source])
        assert manager.resident_bytes() == 0
    else:
        source = C.ArraysCache(1)
        source[0] = mx.ones((2, 4))
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
        forbid(monkeypatch, P, "snapshot_prompt_cache_row")
        assert not coordinate(manager, [source]).store_checkpoint(
            list(range(32)), [source], batch_idx=0
        )
    assert manager.stats.memory_skips >= 1


def test_accounting_and_eviction(memory_manager, monkeypatch):
    cache = NS(
        state={"kv": mx.ones((2, 3)), "nested": [mx.zeros((4,))]},
        meta_state={"offsets": mx.zeros((2,), dtype=mx.int32)},
        prefix_cache_snapshot=lambda: pytest.fail("cloned"),
    )
    assert P._cache_nbytes([cache]) == P._cache_nbytes([cache, cache]) == 48
    manager = memory_manager(budget=1024)
    source = allocated(32)
    runner = coordinate(manager, [source])
    leased = manager.store_kv_blocks(list(range(32)), [source.keys], [source.values])
    assert len(leased) == 2
    manager.release(leased[:1])
    manager.memory_max_bytes = 0
    runner.prepare_prefill(100000)
    assert manager.resident_bytes() == 512
    assert leased[1].ref_cnt == 1
    assert leased[1].keys is not None
    manager.release(leased[1:])
    runner.prepare_prefill(100000)
    assert manager.resident_bytes() == 0
    for base in (object, C._BaseCache):
        cache = type(
            "GrowingCache",
            (base,),
            {"state": mx.ones((1, 256, 1024)), "meta_state": ()},
        )()
        manager = memory_manager(budget=4 << 20)
        monkeypatch.setattr(P, "_clone_prompt_cache_for_apc", lambda cache: cache)
        runner = coordinate(manager, [cache])
        assert manager.store_exact_cache(list(range(18)), [cache])
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 8 << 20)
        runner.prepare_prefill(6001)
        assert manager.resident_bytes() == 0 and not manager._make_room()


def test_single_row_checkpoint(managers):
    tokens = list(range(32))
    arrays = C.ArraysCache(1)
    arrays[0] = mx.ones((1, 3, 5))
    rotating = sample("RotatingKVCache", 8)
    rotating.max_size, rotating.offset, rotating._idx = 8, 32, 4
    caches = [arrays, allocated(32), rotating]
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.__dict__.update(
        uids=[0],
        prompt_cache=caches,
        _right_pad_per_row=None,
        _left_padding_per_row=[0],
        _suffix_lens=[32],
        _processed_prompt_columns=32,
        _apc_mode="exact",
        _apc_manager=managers(blocks=4, block=4),
        _apc_meta=[
            dict(full_input_ids=tokens, prefix_len=0, checkpoint_len=32, extra_hash=0)
        ],
    )
    assert P.extract_prompt_cache_from_batch(caches, 0) is None
    batch._store_apc_exact_checkpoints()
    assert batch._apc_meta[0]["checkpoint_done"] is True
    assert batch._apc_manager.stats.exact_stores == 1


def test_generation_stream_spill(managers, monkeypatch):
    from mlx_vlm.generate.common import generation_stream

    monkeypatch.setenv("APC_MAX_POOL_TENSORS", "1")
    with mx.stream(generation_stream):
        base = mx.arange(64, dtype=mx.float32).reshape(1, 1, 16, 4)
        keys, values = [base + 1, base + 2], [base + 3, base + 4]
    manager = managers("disk", blocks=1)
    assert manager.store_kv_blocks(list(range(16)), keys, values) == []
    manager.disk.flush()
    assert manager.disk.num_blocks_indexed == 1 and manager.disk.disk_bytes > 0


def test_long_prefix_pressure(memory_manager, monkeypatch):
    manager = memory_manager(budget=2 << 20, disk=True)
    coordinator = coordinate(manager, [C.KVCache()])
    manager.disk.queue_max_bytes = 1 << 20
    live_bytes = [0]
    monkeypatch.setattr(
        manager,
        "_memory_headroom",
        lambda: (7 << 20) - manager.resident_bytes() - live_bytes[0],
    )
    for i, length in enumerate([30_000, 30_000, 50_000, 50_000, 100_000]):
        coordinator.prepare_prefill(length)
        cache = allocated(length, i)
        live_bytes[0] = cache.nbytes
        assert manager.store_exact_cache([i] * length, [cache])
        assert manager.resident_bytes() <= manager.memory_max_bytes
        assert manager.disk.pending_bytes <= manager.disk.queue_max_bytes
        del cache
        live_bytes[0] = 0
        mx.clear_cache()
    coordinator.prepare_prefill(100_001)
    assert manager.resident_bytes() == 0
    restored, count = manager.lookup_exact_cache([4] * 100_000 + [99])
    assert count == 100_000
    assert mx.all(restored[0].state[0] == 4).item()
    assert manager.stats_snapshot()["disk_write_failures"] == 0
    assert manager.stats_snapshot()["disk_hits"] == 1


def test_disk_backpressure(tmp_path, monkeypatch):
    disk = P.DiskBlockStore(tmp_path)
    disk.queue_max_bytes = 512
    started, release, second_started = [threading.Event() for _ in range(3)]
    original = disk._write_exact_cache_snapshot

    def slow_write(path, payload):
        if payload.cache_hash == 1:
            started.set()
            assert release.wait(5)
        else:
            second_started.set()
        return original(path, payload)

    monkeypatch.setattr(disk, "_write_exact_cache_snapshot", slow_write)
    caches = [allocated(16), allocated(16, 2)]
    producer = threading.Thread(
        target=lambda: disk.save_exact_cache(2, [2] * 16, 0, caches[1:])
    )
    try:
        disk.save_exact_cache(1, [1] * 16, 0, caches[:1])
        assert started.wait(5)
        assert disk.pending_bytes == 512
        producer.start()
        assert not second_started.wait(0.05)  # Waits for disk, without queue growth.
        assert disk.pending_bytes <= disk.queue_max_bytes
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(5)
        disk.flush()
        disk.close()
    assert not producer.is_alive()
    assert disk.pending_bytes == 0
    assert disk.num_exact_indexed == 2


def test_failed_spill(memory_manager, monkeypatch):
    manager = memory_manager(budget=0, disk=True)

    def fail_write(*args):
        raise OSError("full")

    monkeypatch.setattr(manager.disk, "_write_exact_cache_snapshot", fail_write)
    assert not manager.store_exact_cache(list(range(32)), [allocated(32)])
    stats = manager.stats_snapshot()
    assert stats["exact_stores"] == stats["resident_bytes"] == 0
    assert stats["disk_write_failures"] == 1
    assert not manager.disk._in_flight


@parametrize("synchronous", [True, False])
def test_oversized_write(managers, synchronous):
    disk = managers("disk").disk
    disk.max_bytes = 512
    disk.save_exact_cache(1, [1] * 32, 0, [allocated(32)], synchronous=synchronous)
    disk.flush()
    assert disk.disk_bytes <= disk.max_bytes
    assert disk.num_exact_indexed == 0 and disk.evictions == 1


@parametrize("opt_out", ["environment", "empty_path"])
def test_default_disk_opt_out(tmp_path, monkeypatch, opt_out):
    monkeypatch.setenv("MLX_VLM_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("APC_ENABLED", "1")
    if opt_out == "environment":
        monkeypatch.setenv("APC_DISK_ENABLED", "0")
        overrides = None
    else:
        overrides = {"disk_path": ""}
    manager = P.from_env(overrides=overrides)
    assert manager.disk is None
    assert not (tmp_path / "apc").exists()


@parametrize("tier", ["memory", "disk", "disk-only"])
def test_divergent_dense_prefix(prefix_manager, tier):
    stored = list(range(80))
    divergent = stored[:37] + [999, 998, 997]
    manager = prefix_manager(tier)
    source = C.KVCache()
    keys = mx.array(stored, dtype=mx.float32).reshape(1, 1, -1, 1)
    source.update_and_fetch(keys, keys + 1)
    assert manager.store_exact_cache(stored, [source], extra_hash=7)
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier)
    assert manager.lookup_exact_cache(divergent, extra_hash=8) == (None, 0)
    restored, count = manager.lookup_exact_cache(divergent, extra_hash=7)
    assert count == 32
    assert restored[0].offset == 32
    assert restored[0].state[0].flatten().tolist() == stored[:32]
    assert manager.stats_snapshot()["matched_tokens"] == 32
    if manager.disk:
        assert manager.stats_snapshot()["disk_hits"] == 1
    restored[0].update_and_fetch(mx.full((1, 1, 1, 1), -1), mx.full((1, 1, 1, 1), -2))
    extended, count = manager.lookup_exact_cache(stored + [1000], extra_hash=7)
    assert count == 80
    assert extended[0].state[0].flatten().tolist() == stored
    limited, count = manager.lookup_exact_cache(
        divergent, extra_hash=7, max_prefix_tokens=31
    )
    assert count == limited[0].offset == 16
    assert manager.lookup_exact_cache(
        divergent, extra_hash=7, min_prefix_tokens=32
    ) == (None, 0)


def test_checkpoint_schedule(prefix_manager):
    manager = prefix_manager()
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(cache_model([C.ArraysCache(1)]))
    tokens = list(range(75))
    assert coordinator.checkpoint_lengths(tokens, set()) == [64, 74]
    manager._exact_cache_max = 4
    assert coordinator.checkpoint_lengths(tokens, set()) == [32, 48, 64, 74]
    tokens[45:67] = [999] * 22
    assert coordinator.checkpoint_lengths(tokens, {999}) == [67, 74]
    manager.checkpoint_interval_tokens = 0
    assert coordinator.checkpoint_lengths(tokens, set()) == [74]


def prompt_batch(lm, manager, tokens, prefixes, caches, step=16, **kwargs):
    runner = manager.coordinator(lm)
    suffixes = [ids[n:] for ids, n in zip(tokens, prefixes)]
    padded = [ids + [0] * (max(map(len, suffixes)) - len(ids)) for ids in suffixes]
    return PromptProcessingBatch(
        model=lm,
        uids=list(range(len(tokens))),
        input_ids=suffixes,
        max_tokens=[1] * len(tokens),
        inputs_embeds=embeddings(lm, mx.array(padded)),
        prompt_kwargs={},
        warm_cache=caches,
        prefill_step_size=step,
        apc_manager=manager,
        apc_coordinator=runner,
        apc_meta=[
            dict(
                full_input_ids=ids,
                prefix_len=n,
                checkpoint_lengths=runner.checkpoint_lengths(ids, set()),
            )
            for ids, n in zip(tokens, prefixes)
        ],
        **kwargs,
    )


def finish_batch(batch, sample=None):
    while batch.needs_processing():
        assert batch.prompt_step() > 0
    batch.generate(
        sample or (lambda lp: mx.argmax(lp, axis=-1)),
        [lambda _: False] * len(batch.uids),
    )


def embeddings(lm, tokens):
    return lm.model.embed_tokens(tokens) * getattr(lm.model, "embed_scale", 1)


@parametrize("prefill_step_size", [None, 4, 16, 32])
def test_lfm_padded_prefill(prefix_manager, prefill_step_size):
    mx.random.seed(19)
    lm = language_model("lfm2")
    manager = prefix_manager()
    manager._exact_cache_max = 8
    manager.checkpoint_interval_tokens = 16
    warm_tokens = [i % 50 + 1 for i in range(71)]
    cold_tokens = [i % 30 + 51 for i in range(25)]
    seed_cache = lm.make_cache()
    lm(mx.array([warm_tokens[:64]]), cache=seed_cache)
    assert manager.store_exact_cache(warm_tokens[:64], seed_cache)
    restored, count = manager.lookup_exact_cache(warm_tokens)
    assert count == 64
    caches, _ = warm_exact([restored, lm.make_cache()], [64, 0])
    batch = prompt_batch(
        lm,
        manager,
        [warm_tokens, cold_tokens],
        [64, 0],
        caches,
        prefill_step_size,
        right_pad_per_row=[18, 0],
        suffix_lens=[7, 25],
    )
    sampled = []

    def sample(logprobs):
        sampled.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    finish_batch(batch, sample)
    for row, tokens in enumerate([warm_tokens, cold_tokens]):
        reference = lm.make_cache()
        for start in range(0, len(tokens) - 1, 4):
            lm(
                mx.array([tokens[start : min(start + 4, len(tokens) - 1)]]),
                cache=reference,
            )
        logits = lm(mx.array([tokens[-1:]]), cache=reference).logits[0, -1]
        logprobs = logits - mx.logsumexp(logits)
        assert mx.allclose(sampled[0][row], logprobs, atol=1e-4, rtol=1e-4).item()
        assert mx.argmax(sampled[0][row]).item() == mx.argmax(logits).item()
        assert mx.allclose(
            caches[0][0][row : row + 1], reference[0][0], atol=1e-4, rtol=1e-4
        ).item()


@parametrize("tier", ["memory", "disk"])
def test_diffusion_suffix(prefix_manager, tier):
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig
    from mlx_vlm.tests.test_diffusion_models import (
        FakeProcessor,
        RecordingEncoder,
        tiny_config_dict,
    )

    mx.random.seed(7)
    model = Model(ModelConfig.from_dict(tiny_config_dict()))
    recorder = RecordingEncoder(model.model.encoder)
    model.model.encoder = recorder
    settings = dict(block=2, checkpoint_interval_tokens=4, exact_cache_min_tokens=1)
    manager = prefix_manager(tier, **settings)
    tokens = list(range(2, 13))

    def generate(ids):
        return list(
            stream_generate(
                model,
                FakeProcessor(),
                "",
                input_ids=mx.array([ids]),
                _apc_manager=manager,
                max_tokens=2,
                max_denoising_steps=1,
                _apc_semantic_hash=11,
            )
        )

    generate(tokens)
    assert generate(tokens)[-1].cached_tokens == 10
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier, **settings)
    recorder.input_lengths.clear()
    assert generate(tokens[:9] + [13, 14])[-1].cached_tokens == 8
    assert recorder.input_lengths == [2, 1]


@parametrize("model_name", ["gemma4", "qwen3_5"])
@parametrize("tier", ["memory", "disk"])
@parametrize("path", ["stream", "batch"])
def test_hybrid_prefix_generation(prefix_manager, model_name, tier, path):
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.models.base import InputEmbeddingsFeatures

    mx.random.seed(13)
    lm = language_model(model_name)
    assert P.model_apc_mode(lm) == "exact"
    manager = prefix_manager(tier)
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    tokens = [i % 50 + 1 for i in range(70)] + [51, 52, 53, 54, 55]
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert boundaries == [64, 74]
    ids = mx.array([tokens])
    if path == "stream":
        wrapper = NS(
            language_model=lm,
            get_input_embeddings=lambda ids, *a, **kw: InputEmbeddingsFeatures(
                inputs_embeds=embeddings(lm, ids)
            ),
        )
        list(
            generate_step(
                ids,
                wrapper,
                None,
                None,
                max_tokens=1,
                temperature=0,
                prefill_step_size=16,
                prompt_cache=lm.make_cache(),
                prompt_cache_checkpoint_lengths=boundaries,
                prompt_cache_checkpoint=lambda n, caches: coordinator.store_checkpoint(
                    tokens[:n], caches
                ),
            )
        )
    else:
        finish_batch(prompt_batch(lm, manager, [tokens], [0], lm.make_cache()))
        lengths = sorted(len(e.token_ids) for e in manager._exact_cache.values())
        assert lengths == boundaries
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier)
    divergent = tokens[:70] + [60, 61, 62, 63]
    restored, count = manager.lookup_exact_cache(divergent)
    assert count == 64
    cold_cache = lm.make_cache()
    for start in range(0, count, 16):
        lm(mx.array([divergent[start : start + 16]]), cache=cold_cache)
    suffix = mx.array([divergent[count:]])
    cold = lm(suffix, cache=cold_cache).logits
    warm = lm(suffix, cache=restored).logits
    mx.eval(cold, warm)
    assert mx.allclose(cold, warm, atol=1e-5, rtol=1e-5).item()
    assert mx.array_equal(mx.argmax(cold, axis=-1), mx.argmax(warm, axis=-1)).item()


@parametrize(
    "kind,bits,split",
    [
        ("uniform", 8, False),
        ("turbo", 4.0, False),
        ("turbo", 3.5, False),
        ("turbo", 3.5, True),
    ],
)
def test_packed_disk_and_batch(kind, bits, split, managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    kwargs = {"bits": bits} | ({"key_bits": bits, "value_bits": bits} if split else {})
    source = filled(batch_cache(kind, **kwargs))
    expected = source.extract(0)
    if split:
        assert all(
            isinstance(c, _SplitCodec) for c in (source.key_codec, source.value_codec)
        )
    factory = C.QuantizedKVCache if kind == "uniform" else TurboQuantKVCache
    forbid(monkeypatch, factory, "dequantize_for_apc")
    if kind == "uniform":
        forbid(monkeypatch, mx, "quantize")
    snapshot = snapshot_row([source], 0)
    assert isinstance(snapshot[0], factory)
    same_arrays(snapshot[0].state, expected.state)
    restored, _ = disk_roundtrip(managers, list(range(32)), snapshot)
    assert isinstance(restored[0], factory)
    same_arrays(restored[0].state, snapshot[0].state)
    scheme = "uniform" if kind == "uniform" else "turboquant"
    config = dict(group_size=32, scheme=scheme, **kwargs)
    warm, count = warm_exact([restored, [C.KVCache()]], [32, 0], kv_quant_config=config)
    assert count == 32 and type(warm[0]) is type(source)
    same_arrays(warm[0].extract(0).state, expected.state)
    if kind == "turbo":
        mx.eval(warm[0].update_and_fetch(*kv(1, batch=2)))
        assert warm[0]._idx == 33


def test_quantized_hybrid_snapshots(managers):
    arrays = C.ArraysCache(2)
    arrays.cache = [mx.zeros((1, 32, 32))] * 2
    arrays.left_padding = mx.array([0])
    caches = [arrays, filled(batch_cache("dense")), filled(batch_cache("uniform"))]
    assert all(A.apc_exact_eligible(c) for c in caches)
    cloned = P._clone_cache_entry_for_apc(
        caches[1], min_capacity_tokens=None, eval_targets=[]
    )
    assert isinstance(cloned, C.KVCache) and cloned.offset == 32
    cloned = P._clone_prompt_cache_for_apc(caches)
    assert [type(c) for c in cloned] == [C.ArraysCache, C.KVCache, C.QuantizedKVCache]
    manager = managers()
    assert manager.store_exact_cache(list(range(32)), caches)
    assert manager.stats.exact_stores == 1
    warm, count = manager.lookup_exact_cache(list(range(32)) + [999])
    assert count == 32 and len(warm) == 3


@parametrize("scheme", ["uniform", "turboquant"])
def test_warm_cache_quantization_policy(scheme, managers):
    bits = 8 if scheme == "uniform" else 3.5
    config = dict(bits=bits, group_size=32, scheme=scheme)
    manager = managers(blocks=32)
    tokens = list(range(32 if scheme == "uniform" else 16))
    blocks = store_blocks(manager, tokens, layers=4, dim=32)
    manager.release(blocks)
    blocks, count = manager.lookup_prefix(tokens)
    assert count == len(tokens)
    live = _make_cache(
        NS(layers=[NS()] * 4),
        [0],
        kv_bits=float(bits),
        kv_group_size=32,
        kv_quant_scheme=scheme,
    )
    for cache in live:
        filled(cache, len(tokens))
    if scheme == "uniform":
        warm, count = warm_blocks(
            [{"matched_blocks": blocks, "prefix_len": len(tokens)}, None],
            num_layers=4,
            kv_quant_config=config,
        )
        assert count == len(tokens)
        assert warm[-1].left_padding.tolist() == [0, len(tokens)]
        single = P.make_warm_kv_cache(
            blocks, kv_quant_config={"bits": 8.0, "group_size": 32.0}
        )
        assert isinstance(single[0], C.QuantizedKVCache)
        assert (single[0].bits, single[0].group_size) == (8, 32)

    else:
        warm = P.make_warm_batch_kv_cache(blocks, kv_quant_config=config)
        assert isinstance(warm[0], BatchTurboQuantKVCache)
    assert [type(c) for c in warm] == [type(c) for c in live]
    assert isinstance(warm[-1], C.BatchKVCache)
    manager.release(blocks)
    if scheme == "uniform":
        arrays = C.ArraysCache(2)
        arrays.cache = [mx.zeros((1, 16, 32))] * 2
        row = [arrays] + [filled(C.KVCache(), 16) for _ in range(3)]
        live_arrays = clone(arrays)
        live_arrays.left_padding = mx.array([0])
        live = [live_arrays] + [
            filled(
                batch_cache("uniform" if C.should_quantize_kv_layer(i, 4) else "dense"),
                16,
            )
            for i in range(1, 4)
        ]
        warm, _ = warm_exact([row], [16], kv_quant_config=config)
        assert [type(c) for c in warm] == [type(c) for c in live]
        extended = _extend_cache(live, warm)
        assert extended[1].offset.shape[0] == 2
        assert isinstance(extended[1], C.BatchQuantizedKVCache)
        assert isinstance(extended[-1], C.BatchKVCache)


def test_short_and_multimodal_prefixes(managers):
    config = NS(model_type="deepseek_v4", vision_n_layers=32, vocab_size=129280)
    assert P.multimodal_token_ids_from_config(config) == set(range(129280, 129285))
    assert (
        P.adjust_prefix_to_text_suffix_boundary(
            [1, 42, 42], desired_prefix_len=1, media_token_ids={42}, max_prefix_tokens=2
        )
        == 0
    )
    cache = C.ArraysCache(1)
    cache[0] = mx.zeros((1, 2, 1, 32))
    manager = managers()
    manager.store_exact_cache([1], [cache])
    assert manager.lookup_exact_cache(list(range(1, 400))) == (None, 0)


# TurboQuant

# Cache codecs and integration


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


def test_turboquant_prod_is_nearly_unbiased_across_seeds():
    mx.random.seed(42)
    keys = _sample_unit_vectors(128, 64)
    queries = mx.random.normal((128, 64))
    true_inner_products = mx.sum(keys * queries, axis=-1)

    estimates = []
    for seed in range(16):
        codec = _TurboQuantProdCodec(64, 2, seed=seed)
        state = codec.quantize(keys)
        reconstructed = codec.dequantize(state)
        estimates.append(mx.sum(reconstructed * queries, axis=-1))

    mean_estimate = mx.mean(mx.stack(estimates), axis=0)
    bias = mx.mean(mean_estimate - true_inner_products).item()
    assert abs(bias) < 0.03


def test_turboquant_skips_non_kv_cache_entries():
    linear_cache = ArraysCache(size=2)
    linear_cache[0] = mx.zeros((1, 8))
    linear_cache[1] = mx.ones((1, 8))

    attention_cache = KVCache()
    attention_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
    )
    prompt_cache = [linear_cache, attention_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], ArraysCache)
    assert isinstance(prompt_cache[1], TurboQuantKVCache)


def test_batch_turboquant_filter_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    cache = BatchTurboQuantKVCache([0], bits=3.5)

    cache.update_and_fetch(keys, values)
    cache.filter(mx.array([0]))

    assert cache.offset.tolist() == [3]
    assert cache.left_padding.tolist() == [0]


def test_batch_turboquant_cache_supports_uniform_right_trim():
    cache = BatchTurboQuantKVCache([0, 1], bits=3.5)
    keys = mx.ones((2, 2, 5, 8), dtype=mx.float16)
    values = mx.ones((2, 2, 5, 8), dtype=mx.float16)

    cache.update_and_fetch(keys, values)
    trimmed = cache.trim(2)

    assert trimmed == 2
    assert cache.is_trimmable()
    assert cache._idx == 3
    assert cache.offset.tolist() == [3, 2]
    state_keys, state_values, offset, left_padding = cache.state
    assert state_keys.norms.shape[2] == 3
    assert state_values.norms.shape[2] == 3
    assert offset.tolist() == [3, 2]
    assert left_padding.tolist() == [0, 1]


def test_batch_turboquant_extend_pads_shorter_uniform_batch():
    longer = BatchTurboQuantKVCache([0], bits=3.5)
    shorter = BatchTurboQuantKVCache([0], bits=3.5)

    longer.update_and_fetch(
        mx.ones((1, 2, 5, 8), dtype=mx.float16), mx.ones((1, 2, 5, 8), dtype=mx.float16)
    )
    shorter.update_and_fetch(
        mx.ones((1, 2, 3, 8), dtype=mx.float16), mx.ones((1, 2, 3, 8), dtype=mx.float16)
    )
    longer.extend(shorter)

    assert longer.offset.tolist() == [5, 3]
    assert longer.left_padding.tolist() == [0, 2]
    assert longer._idx == 5


def test_turboquant_decode_attention_4bit_uses_paper_prod_key_codec():
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    # Keys now use MSE-only codec (QJL/Prod dropped for speed+quality)
    assert type(turbo_cache.key_codec).__name__ == "_TurboQuantMSECodec"
    output = scaled_dot_product_attention(
        queries, turbo_keys, turbo_values, turbo_cache, scale=32**-0.5, mask=None
    )

    assert output.shape == queries.shape


def _assert_mse_states_equal(left, right):
    assert bool(mx.all(left.norms == right.norms).item())
    assert bool(mx.all(left.indices == right.indices).item())


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_turboquant_mse_prefill_decode_matches_batch_quantized_state(dim):
    mx.random.seed(99)
    keys = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)
    values = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)

    batch_cache = TurboQuantKVCache(bits=4.0)
    batch_cache.update_and_fetch(keys, values)

    split_cache = TurboQuantKVCache(bits=4.0)
    split_cache.update_and_fetch(keys[:, :, :16, :], values[:, :, :16, :])
    split_cache.update_and_fetch(keys[:, :, 16:17, :], values[:, :, 16:17, :])

    batch_keys, batch_values = batch_cache.state
    split_keys, split_values = split_cache.state
    _assert_mse_states_equal(batch_keys, split_keys)
    _assert_mse_states_equal(batch_values, split_values)


def test_turboquant_prefill_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 12, 32))
    values = mx.random.normal((1, 2, 12, 32))
    queries = mx.random.normal((1, 4, 4, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys, turbo_values
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask="causal",
    )
    quantized = scaled_dot_product_attention(
        queries, turbo_keys, turbo_values, turbo_cache, scale=32**-0.5, mask="causal"
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4


def test_resolve_kv_bits_validates_overrides():
    with pytest.raises(ValueError):
        resolve_kv_bits(4, 0.5, None)
    with pytest.raises(ValueError):
        resolve_kv_bits(4, None, 3.25)


def test_asymmetric_attention_matches_dequantized_reference():
    cache = TurboQuantKVCache(bits=4, key_bits=8, value_bits=3)
    keys = mx.random.normal((1, 4, 128, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 128, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)

    queries = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
    scale = 256**-0.5
    out = cache.quantized_attention(queries, scale=scale, mask=None)

    deq_keys, deq_values = cache.dequantize()
    reference = scaled_dot_product_attention(
        queries, deq_keys, deq_values, cache=None, scale=scale, mask=None
    )
    error = mx.sqrt(
        mx.sum((out.astype(mx.float32) - reference.astype(mx.float32)) ** 2)
        / mx.sum(reference.astype(mx.float32) ** 2)
    ).item()
    assert error < 0.02


@pytest.mark.parametrize(
    "mixed", [False, True], ids=["turboquant-defaults", "mixed-schemes"]
)
def test_maybe_quantize_kv_cache_policy(mixed):
    from mlx_vlm.turboquant import HybridQuantKVCache

    options = (
        dict(
            kv_bits=8,
            kv_quant_scheme="uniform",
            kv_value_bits=3,
            kv_value_scheme="turboquant",
        )
        if mixed
        else dict(kv_bits=3.5, kv_quant_scheme="turboquant")
    )
    prompt_cache = [KVCache() for _ in range(3)]
    maybe_quantize_kv_cache(
        prompt_cache, quantized_kv_start=0, kv_group_size=64, **options
    )
    kind = HybridQuantKVCache if mixed else TurboQuantKVCache
    converted = [entry for entry in prompt_cache if isinstance(entry, kind)]
    assert converted
    for entry in converted:
        if mixed:
            assert (entry.policy.key.scheme, entry.policy.key.bits) == ("uniform", 8.0)
            assert (entry.policy.value.scheme, entry.policy.value.bits) == (
                "turboquant",
                3.0,
            )
        else:
            assert (entry.key_bits, entry.value_bits) == (3.0, 4.0)


def _turbo_quant_config(**overrides):
    config = {"bits": 3.5, "group_size": 64, "scheme": "turboquant"}
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "mixed", [False, True], ids=["turboquant-bits", "mixed-schemes"]
)
def test_apc_stream_warm_cache_policy(mixed):
    from mlx_vlm.apc import _fill_stream_layer_cache
    from mlx_vlm.turboquant import HybridQuantKVCache

    config = (
        dict(
            bits=8,
            group_size=64,
            scheme="uniform",
            value_bits=3,
            value_scheme="turboquant",
        )
        if mixed
        else _turbo_quant_config(key_bits=8, value_bits=3)
    )
    built = _fill_stream_layer_cache(
        *[mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16) for _ in range(2)],
        prefix_len=8,
        quantize=True,
        kv_quant_config=config,
    )
    if mixed:
        assert isinstance(built, HybridQuantKVCache)
        assert built.policy.key.scheme == "uniform"
        assert built.policy.value.scheme == "turboquant"
        assert built.offset == 8
    else:
        assert (built.key_bits, built.value_bits) == (8.0, 3.0)


def test_apc_warm_cache_defaults_match_live_split():
    from mlx_vlm.apc import _empty_quant_batch_cache

    built = _empty_quant_batch_cache([0], _turbo_quant_config())
    assert (built.key_bits, built.value_bits) == (3.0, 4.0)


def test_kv_quant_policy_round_trips_through_config():
    from mlx_vlm.kv_quant import from_config, from_legacy

    for args in [(3.5, "turboquant", 64, None, None), (3.5, "turboquant", 64, 8, 3)]:
        policy = from_legacy(*args)
        assert from_config(policy.to_config()) == policy


def test_kv_quant_policy_uniform_and_none():
    from mlx_vlm.kv_quant import from_legacy

    assert from_legacy(None) is None
    uniform = from_legacy(8, "uniform", 64)
    assert uniform.scheme == "uniform"
    assert not uniform.is_turboquant
    assert (uniform.key.bits, uniform.value.bits) == (8.0, 8.0)
    assert uniform.is_homogeneous


def test_kv_quant_policy_scheme_property_rejects_heterogeneous():
    from mlx_vlm.kv_quant import from_legacy

    policy = from_legacy(
        8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant"
    )
    with pytest.raises(ValueError):
        policy.scheme


@pytest.mark.parametrize(
    "bits,scheme,overrides",
    [
        (8, "uniform", dict(kv_key_scheme="turbo3")),
        (3.5, "turboquant", dict(kv_key_bits=3.5, kv_key_scheme="uniform")),
    ],
    ids=["unknown-scheme", "fractional-uniform"],
)
def test_invalid_kv_quant_policy(bits, scheme, overrides):
    from mlx_vlm.kv_quant import from_legacy

    with pytest.raises(ValueError):
        from_legacy(bits, scheme, 64, **overrides)


def test_kv_quant_heterogeneous_round_trips_and_fingerprints_distinctly():
    from mlx_vlm.kv_quant import from_config, from_legacy

    hetero = from_legacy(
        8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant"
    )
    assert from_config(hetero.to_config()) == hetero

    homogeneous = from_legacy(8, "uniform", 64)
    assert homogeneous.fingerprint(0) == "kv8.0-64-uniform-0"
    assert hetero.fingerprint(0) != homogeneous.fingerprint(0)
    assert hetero.fingerprint(0).endswith("-ksuniform-vsturboquant")


def _hybrid_policy():
    from mlx_vlm.kv_quant import from_legacy

    return from_legacy(8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant")


def test_hybrid_cache_appends_across_decode_steps():
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    cache.update_and_fetch(
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
    )
    for _ in range(3):
        deq_keys, deq_values = cache.update_and_fetch(
            mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16),
            mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16),
        )
    assert cache.offset == 11
    assert deq_keys.shape == (1, 4, 11, 256)
    assert deq_values.shape == (1, 4, 11, 256)


def test_hybrid_cache_meta_state_round_trips_policy():
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    cache.offset = 12
    restored = HybridQuantKVCache(from_legacy(4, "uniform", 64))
    restored.meta_state = cache.meta_state
    assert restored.policy == cache.policy
    assert restored.offset == 12
    assert restored.seed == cache.seed


def test_batch_generator_accepts_scheme_overrides():
    import inspect

    from mlx_vlm.generate.ar import BatchGenerator, _make_cache

    for target in (BatchGenerator.__init__, _make_cache):
        params = inspect.signature(target).parameters
        assert "kv_key_scheme" in params
        assert "kv_value_scheme" in params


@pytest.mark.parametrize(
    "builder,message", [("live", "batch path"), ("prefix", "batch prefix caches")]
)
def test_batch_cache_rejects_mixed_schemes(builder, message):
    from mlx_vlm.apc import _empty_quant_batch_cache

    with pytest.raises(NotImplementedError, match=message):
        if builder == "live":
            _make_cache(
                NS(make_cache=lambda: [KVCache()]),
                [0],
                kv_bits=8,
                kv_quant_scheme="uniform",
                kv_value_bits=3,
                kv_value_scheme="turboquant",
            )
        else:
            _empty_quant_batch_cache(
                [0],
                dict(
                    bits=8,
                    group_size=64,
                    scheme="uniform",
                    value_bits=3,
                    value_scheme="turboquant",
                ),
            )


def test_hybrid_cache_trims_fractional_turboquant_tensor():
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache, _SplitCodec

    cache = HybridQuantKVCache(
        from_legacy(8, "uniform", 64, kv_value_bits=3.5, kv_value_scheme="turboquant")
    )
    cache.update_and_fetch(
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
    )
    assert isinstance(cache.value_quantizer.codec, _SplitCodec)
    assert cache.trim(5) == 5
    assert cache.offset == 15
    deq_keys, deq_values = cache.dequantize()
    assert deq_keys.shape[-2] == 15
    assert deq_values.shape[-2] == 15


# Batched attention

TURBO_HEADS, TURBO_DIM = 4, 64  # kv heads, head_dim
TURBO_BITS = 4
TURBO_SCALE = TURBO_DIM**-0.5


def _filled(left_padding, seq_len, batch=None, bits=TURBO_BITS):
    batch = len(left_padding) if batch is None else batch
    cache = BatchTurboQuantKVCache(left_padding, bits=bits)
    keys, values = cache.update_and_fetch(*_rand_kv(batch, seq_len))
    return cache, keys, values


class TestFusedPathGuard:
    def test_cached_eligibility_tracks_batch_lifecycle(self):
        cache = BatchTurboQuantKVCache([0], bits=TURBO_BITS)
        other = BatchTurboQuantKVCache([0], bits=TURBO_BITS)
        assert cache.fused_attention_eligible

        cache.extend(other)
        assert not cache.fused_attention_eligible
        assert not _turboquant_attention_applies(cache)

        cache.filter(mx.array([0]))
        assert cache.fused_attention_eligible
        assert _turboquant_attention_applies(cache)

        cache.state = BatchTurboQuantKVCache([2], bits=TURBO_BITS).state
        assert not cache.fused_attention_eligible
        assert not _turboquant_attention_applies(cache)


class TestNumericalEquivalence:
    """The fused path must agree with the dequantizing fallback."""

    def _reference(self, cache, queries, keys, values, mask=None):
        dq_k, dq_v = cache.dequantize(keys, values)
        return mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=TURBO_SCALE,
            mask=mask,
        )

    def test_multi_row_still_produces_correct_shape(self):
        cache, keys, values = _filled([0, 0], 12)
        queries = mx.random.normal((2, TURBO_HEADS, 1, TURBO_DIM))
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=TURBO_SCALE, mask=None
        )
        mx.eval(out)
        assert out.shape == (2, TURBO_HEADS, 1, TURBO_DIM)


class TestDecodeMemoryIsFlat:
    """Regression guard for the bug this change fixes.

    The dequantizing fallback materialised the whole KV cache as float32 on
    every step, so peak memory scaled with the context length. The fused path
    reads the quantized state in place.
    """

    def _peak_delta_for(self, seq_len):
        cache, keys, values = _filled([0], seq_len)
        queries = mx.random.normal((1, TURBO_HEADS, 1, TURBO_DIM))
        mx.eval(cache.keys, cache.values, queries)
        mx.clear_cache()

        before = mx.get_peak_memory()
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=TURBO_SCALE, mask=None
        )
        mx.eval(out)
        return mx.get_peak_memory() - before

    def test_peak_does_not_scale_with_context(self):
        short = self._peak_delta_for(256)
        long = self._peak_delta_for(4096)
        # 16x the context. Dequantizing would grow the step's peak roughly in
        # step with it; the fused kernels keep it bounded.
        assert long <= max(short, 1 << 20) * 4


# Rotated value kernels


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


@pytest.mark.parametrize("kind,batch_size", [("turbo", 1), ("batch", 2), ("hybrid", 1)])
@pytest.mark.parametrize("bits", [3, 3.5])
def test_packed_memory_profiles_include_spare_capacity(kind, batch_size, bits):
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache, _state_nbytes

    if kind == "hybrid":
        cache = HybridQuantKVCache(
            from_legacy(
                8, "uniform", 64, kv_value_bits=bits, kv_value_scheme="turboquant"
            )
        )
    elif kind == "batch":
        cache = BatchTurboQuantKVCache([0, 3], bits)
    else:
        cache = TurboQuantKVCache(bits)

    def advance(length):
        cache.update_and_fetch(
            mx.ones((batch_size, 1, length, 64)),
            mx.ones((batch_size, 1, length, 128)),
        )

    empty = cache_memory_components([cache], 0)[0]
    assert not empty.fallback and empty.footprint(6000) == 0
    advance(1)
    if kind == "turbo":
        cache = clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=[])
    profile = cache_memory_components([cache], 1, batch_size=batch_size)[0]
    assert not profile.fallback
    for start in range(1, 6000, 257):
        advance(min(257, 6000 - start))
    allocated = _state_nbytes(cache.keys) + _state_nbytes(cache.values)
    estimate = batch_size * profile.footprint(6000, 257)
    assert allocated <= estimate < 1.1 * allocated
    current = cache_memory_components([cache], 6000, batch_size=batch_size)[0]
    assert current.source_bytes * batch_size == allocated
    if kind != "hybrid":
        assert allocated > cache.nbytes


@parametrize("mode", ["block", "exact"])
@parametrize("tier", ["memory", "disk"])
@parametrize("restore", ["single", "batch", "failed-batch"])
def test_restored_tokens_count_successful_restores(
    mode, tier, restore, managers, monkeypatch
):
    manager = managers(tier, namespace="stats")
    tokens = list(range(32))
    row = [sample("ArraysCache", 32), sample("KVCache", 32)]
    if mode == "exact":
        assert manager.store_exact_cache(tokens, row)
    else:
        manager.release(manager.store_kv_blocks(tokens, [row[1].keys], [row[1].values]))
    stats = manager.stats_snapshot()
    assert stats["restored_tokens"] == 0
    assert stats["stored_tokens"] == (32 if mode == "block" else 0)
    if tier == "disk":
        manager.close()
        manager = managers(tier, namespace="stats")
    caches = [ArraysCache(2), KVCache()] if mode == "exact" else [KVCache()]
    runner = coordinate(manager, caches)
    hit = runner.lookup(
        tokens + [99],
        extra_hash=0,
        safe_lookup_min=0,
        suffix_is_text_only=lambda _: True,
        prefix_has_media=lambda _: False,
    )
    assert hit is not None
    before = manager.stats_snapshot()
    assert before["matched_tokens"] > 0 and before["restored_tokens"] == 0
    if restore == "single":
        caches = runner.materialize_single(hit, min_capacity_tokens=33)
    else:
        if restore == "failed-batch":
            builder = "exact" if mode == "exact" else "kv"
            monkeypatch.setattr(
                P, f"make_warm_batch_{builder}_cache_multi", lambda *a, **kw: (None, 0)
            )
        caches, _ = runner.merge_rows([hit, None], [hit["prefix_len"], 0])
    runner.release_hit(hit)
    stats = manager.stats_snapshot()
    assert (caches is None) == (restore == "failed-batch")
    assert stats["restored_tokens"] == (0 if caches is None else hit["prefix_len"])
    assert stats["token_hit_rate"] == before["token_hit_rate"]
    if tier == "disk":
        assert stats["disk_hits"] > 0
    manager.reset_stats()
    assert manager.stats_snapshot()["restored_tokens"] == 0
    assert manager.stats_snapshot()["stored_tokens"] == 0


def test_a_single_restored_row_keeps_its_own_cache(managers):
    """One restored row gets the hit's own cache back, the object
    materialize_single restores on the single-sequence path, not a batch
    merge of it. Models with a single-row shortcut (qwen3_5) otherwise
    extract() and re-merge() the full KV on every decode step. Two rows
    still merge."""
    manager = managers()
    tokens = list(range(32))
    assert manager.store_exact_cache(
        tokens, [sample("ArraysCache", 32), sample("KVCache", 32)]
    )
    runner = coordinate(manager, [ArraysCache(2), KVCache()])

    def hit():
        return runner.lookup(
            tokens + [99],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _: True,
            prefix_has_media=lambda _: False,
        )

    one = hit()
    caches, prefix = runner.merge_rows([one], [one["prefix_len"]])
    assert caches is one["warm_cache"] and prefix == one["prefix_len"]
    assert manager.stats_snapshot()["restored_tokens"] == one["prefix_len"]
    runner.release_hit(one)

    two = hit()
    caches, _ = runner.merge_rows([two, None], [two["prefix_len"], 0])
    assert caches is not two["warm_cache"]
    assert any(type(c).__name__.startswith("Batch") for c in caches)
    runner.release_hit(two)
