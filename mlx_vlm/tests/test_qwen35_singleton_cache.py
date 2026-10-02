"""Preserve allocation when a Qwen3.5 batch shrinks to one request."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models.cache import BatchKVCache, KVCache
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.config import TextConfig


def model():
    mx.random.seed(7)
    return language.LanguageModel(
        TextConfig(
            model_type="qwen3_5_text",
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            full_attention_interval=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=2048,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 1, 0],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
            },
        ),
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=29,
            video_token_id=30,
            vision_start_token_id=31,
        ),
    )


def assert_close(actual, expected):
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("survivor", [0, 1])
def test_shrinking_batch_preserves_capacity_and_matches_regular_cache(survivor):
    lm = model()
    seeds = [lm.make_cache(), lm.make_cache()]
    for cache, tokens in zip(seeds, ([1, 2, 3, 4], [2, 3])):
        mx.eval(lm(mx.array([tokens]), cache=cache).logits)
    reference_cache = copy.deepcopy(seeds[survivor])
    batch = [type(a).merge([a, b]) for a, b in zip(*seeds)]
    mx.eval(lm(mx.array([[5], [6]]), cache=batch).logits)
    mx.eval(lm(mx.array([[5 + survivor]]), cache=reference_cache).logits)
    for entry in batch:
        entry.filter(mx.array([survivor]))
    attention = batch[lm.model.fa_idx]
    assert type(attention) is BatchKVCache
    assert attention.left_padding.tolist() == [0]
    capacities = []
    for step in range(270):
        token = mx.array([[7 + step % 5]])
        expected = lm(token, cache=reference_cache).logits
        actual = lm(token, cache=batch).logits
        assert_close(actual, expected)
        # Keep the batch object and its spare storage through decode and growth.
        assert batch[lm.model.fa_idx] is attention
        assert attention._idx == reference_cache[lm.model.fa_idx].offset
        assert attention.offset.tolist() == [attention._idx]
        assert_close(
            attention.keys[:, :, : attention._idx],
            reference_cache[lm.model.fa_idx].state[0],
        )
        assert_close(
            attention.values[:, :, : attention._idx],
            reference_cache[lm.model.fa_idx].state[1],
        )
        capacities.append(attention.keys.shape[2])
    assert len(set(capacities)) <= 3
    assert capacities[-1] > attention._idx

    # A new request must still join the surviving batch after borrowed storage
    # has grown. Compare both rows with their independent ordinary caches.
    joining = lm.make_cache()
    mx.eval(lm(mx.array([[2, 4, 6]]), cache=joining).logits)
    joining_reference = copy.deepcopy(joining)
    for live, new in zip(batch, joining):
        live.extend(type(new).merge([new]))
    for step in range(3):
        first, second = mx.array([[8 + step]]), mx.array([[12 + step]])
        expected_first = lm(first, cache=reference_cache).logits
        expected_second = lm(second, cache=joining_reference).logits
        actual = lm(mx.concatenate([first, second]), cache=batch).logits
        assert_close(actual, mx.concatenate([expected_first, expected_second]))


def test_borrow_retains_backing_arrays_and_used_offset():
    cache = BatchKVCache([0])
    cache.update_and_fetch(mx.ones((1, 2, 7, 16)), mx.ones((1, 2, 7, 16)))
    row = language._borrow_singleton_kv_cache(cache)
    assert type(row) is KVCache
    assert row.keys is cache.keys and row.values is cache.values
    assert row.offset == 7
    assert row.keys.shape[2] > row.offset


@pytest.mark.parametrize(
    "kind", ["empty", "padded", "right_padding", "multirow", "subclass"]
)
def test_borrow_rejects_other_cache_layouts(kind):
    class CustomBatch(BatchKVCache):
        pass

    cls = CustomBatch if kind == "subclass" else BatchKVCache
    batch = 2 if kind == "multirow" else 1
    cache = cls([2 if kind == "padded" else 0] * batch)
    if kind != "empty":
        cache.update_and_fetch(mx.ones((batch, 2, 7, 16)), mx.ones((batch, 2, 7, 16)))
    if kind == "right_padding":
        cache.prepare(right_padding=[1])
    assert language._borrow_singleton_kv_cache(cache) is None


def test_prefill_and_capture_keep_existing_path(monkeypatch):
    lm = model()
    seed = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3]]), cache=seed).logits)

    def unexpected_borrow(entry):
        raise AssertionError("borrowed outside ordinary single-token decode")

    monkeypatch.setattr(language, "_borrow_singleton_kv_cache", unexpected_borrow)
    for tokens, capture in (([[4, 5]], None), ([[4]], [0])):
        batch = [type(c).merge([copy.deepcopy(c)]) for c in seed]
        result = lm(mx.array(tokens), cache=batch, capture_layer_ids=capture)
        mx.eval(result.logits)
        assert mx.all(mx.isfinite(result.logits)).item()
