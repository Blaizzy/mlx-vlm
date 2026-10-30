import json
import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.mistral_large4.config import TextConfig
from mlx_vlm.models.mistral_large4.language import (
    LanguageModel,
    LatentCache,
    NVFP4SwitchLinear,
    SoftmaxScaling,
    rotate,
)


def tiny_config(**kwargs):
    config = dict(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        vocab_size=50,
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3],
        indexer_types=[
            "full",
            "full",
            "shared",
            "full",
            "full",
            "shared",
            "full",
            "shared",
        ],
    )
    config.update(kwargs)
    return TextConfig(**config)


def test_mistral_large4_cached_dsa_and_shared_kv_match_full_sequence():
    mx.random.seed(17)
    model = LanguageModel(tiny_config())
    model.eval()
    ids = mx.array([[1, 2, 7, 5, 9, 13, 11], [2, 4, 6, 8, 10, 12, 14]])
    expected = model(ids).logits
    cache = model.make_cache()
    chunks = [
        model(ids[:, a:b], cache=cache).logits for a, b in [(0, 3), (3, 5), (5, 7)]
    ]
    actual = mx.concatenate(chunks, axis=1)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5
    )
    assert all(c.offset == 7 for c in cache)
    assert cache[7].keys is None  # This layer reuses layer 3's compressed KV.
    assert cache[3].keys.shape == (2, 1, 7, 16)
    assert cache[0].index_keys.shape == (2, 7, 16)
    assert cache[2].index_keys is None  # Indexer sharing carries only indices.


def test_mistral_large4_cache_restore_and_trim():
    model = LanguageModel(tiny_config())
    model.eval()
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    cache = model.make_cache()
    model(ids, cache=cache)
    restored = [LatentCache.from_state(c.state, c.meta_state) for c in cache]
    for c in restored:
        assert c.trim(2) == 2
    actual = model(ids[:, 4:], cache=restored).logits
    expected = model(ids).logits[:, 4:]
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=3e-6, rtol=3e-5
    )


def test_mistral_large4_rotary_pair_layout_and_position_scaling():
    x = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos, sin = mx.array([[[[0.0, 1.0]]]]), mx.array([[[[1.0, 0.0]]]])
    np.testing.assert_array_equal(
        np.array(rotate(x, cos, sin)), [[[[-2.0, 3.0, 1.0, 4.0]]]]
    )
    scaling = SoftmaxScaling(2)
    scaling.s = mx.array([0.25, -0.5])
    scaling.b = mx.array([1.0, 2.0])
    positions = mx.array([[0, 3]])
    actual = scaling(mx.ones((1, 2, 2, 3)), positions)
    expected = np.array(
        [[[1.0, 1.0 + 0.25 * math.log(4)], [2.0, 2.0 - 0.5 * math.log(4)]]]
    )[..., None]
    np.testing.assert_allclose(
        np.array(actual), np.broadcast_to(expected, (1, 2, 2, 3)), atol=1e-6
    )


def test_mistral_large4_nvfp4_global_scales_preserve_weight_values():
    # Nibble 2 is FP4 1.0, E4M3 byte 56 is 1.0; global scales differ by expert.
    linear = NVFP4SwitchLinear(16, 4, 2)
    linear.weight = mx.full((2, 4, 2), 0x22222222, mx.uint32)
    linear.scales = mx.full((2, 4, 1), 56, mx.uint8)
    linear.global_scale = mx.array([0.25, 2.0], mx.float32)
    x = mx.ones((2, 1, 16), mx.bfloat16)
    actual = linear(x, mx.array([0, 1]))
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), [[[4.0] * 4], [[32.0] * 4]]
    )


def test_mistral_large4_partitions_allow_three_ranks_and_reject_wrong_shards():
    class Group:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

        def size(self):
            return 3

    model = LanguageModel(tiny_config())
    original = model.layers[1].feed_forward.experts.w1.weight
    model.shard(Group(1))
    moe = model.layers[1].feed_forward
    assert (moe.start, moe.end) == (2, 5)
    np.testing.assert_array_equal(
        np.array(moe.experts.w1.weight), np.array(original[2:5])
    )
    with pytest.raises(ValueError, match="partition"):
        model.shard(Group(2))


def test_mistral_large4_invalid_kv_sharing_fails_at_configuration():
    with pytest.raises(ValueError, match="rotary layout"):
        tiny_config(kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 0])
    with pytest.raises(ValueError, match="Invalid KV source"):
        tiny_config(kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 8])


def test_mistral_large4_native_config_keeps_generation_defaults(tmp_path, monkeypatch):
    from mlx_vlm.models.mistral_large4 import config
    from mlx_vlm.utils import load_config

    (tmp_path / "params.json").write_text(
        json.dumps({"lerope": True, "ssmax_scaling": True, "dsa": {"topk": 4}})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 3], "temperature": 0.7})
    )
    monkeypatch.setattr(
        config,
        "config_from_params",
        lambda params: {"model_type": "mistral_large4", "eos_token_id": 2},
    )
    result = load_config(tmp_path)
    assert result["_native_mistral_large4"]
    assert result["eos_token_id"] == [2, 3]
    assert result["temperature"] == 0.7
