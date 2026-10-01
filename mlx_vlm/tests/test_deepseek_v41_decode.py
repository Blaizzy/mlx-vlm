"""Aligned decode cache transitions and device-only expert routing."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_vlm.models.deepseek_v41 import fakequant
from mlx_vlm.models.deepseek_v41.engram import NgramHashState
from mlx_vlm.models.deepseek_v41.language import (
    BatchDeepseekV41Cache,
    DeepseekV41Cache,
    DeepseekV41MoE,
    LanguageModel,
)
from mlx_vlm.tests.test_deepseek_v41_sharding import small_config


class RowCache(BatchDeepseekV41Cache):
    def for_decode(self, width):
        return self


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_aligned_decode_then_filter_join_and_prefill(dtype, monkeypatch):
    # Isolate cache transitions from FP32 matmul changes crossing QAT rounding
    # thresholds. The BF16 case exercises the model's real fake quantization.
    monkeypatch.setattr(fakequant, "DISABLE", dtype == mx.float32)
    mx.random.seed(31)
    config = small_config()
    model = LanguageModel(config)
    model.head.weight = mx.random.normal(model.head.weight.shape) * 0.05
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    model.head.weight = model.head.weight.astype(mx.float32)
    model.engram_hash = NgramHashState(
        config, model.layout, token_map=[i % 7 for i in range(config.vocab_size)]
    )
    fast = BatchDeepseekV41Cache(
        [DeepseekV41Cache(len(model.layers)) for _ in range(4)]
    )
    slow = RowCache([DeepseekV41Cache(len(model.layers)) for _ in range(4)])

    def compare(tokens):
        tokens = tokens % config.vocab_size
        a = model(tokens, cache=[fast]).logits
        b = model(tokens, cache=[slow]).logits
        mx.eval(a, b, fast.state, slow.state)
        assert mx.allclose(
            a,
            b,
            atol=0.02 if dtype == mx.bfloat16 else 1e-4,
            rtol=0.02 if dtype == mx.bfloat16 else 1e-4,
        ).item()
        assert fast.offset.tolist() == slow.offset.tolist()
        assert fast.size() == slow.size()
        for i in range(tokens.shape[0]):
            assert mx.array_equal(fast.extract(i).engram, slow.extract(i).engram).item()

    compare(mx.array([[3 + row, 7, 9, 11, 13, 15, 17] for row in range(4)]))
    assert fast._batched is None
    native = None
    for step in range(9):
        compare(mx.array([[19 + row + step] for row in range(4)]))
        native = native or fast._batched
        assert fast._batched is native
        assert fast.memory_profile(fast.size()).source_bytes > 0
    snapshot = fast.extract(2)
    saved = [mx.array(a) for a in snapshot.state]
    fast.filter([2, 0])
    slow.filter([2, 0])
    assert fast._batched is None
    compare(mx.array([[41], [43]]))
    assert fast._batched is not None
    for a, b in zip(saved, snapshot.state):
        assert mx.array_equal(a, b).item()

    # A joining request with a different compression phase must remain ragged.
    joined = model.make_cache()
    mx.eval(model(mx.array([[11, 13, 15]]), cache=joined).logits)
    fast.extend(BatchDeepseekV41Cache([joined[0].extract(0)]))
    slow.extend(RowCache([joined[0].extract(0)]))
    compare(mx.array([[45], [47], [49]]))
    assert fast._batched is None
    fast.filter([1, 0])
    slow.filter([1, 0])
    compare(mx.array([[51], [53]]))
    assert fast._batched is not None
    # A chunk after decode unpacks current buffers before entering the row path.
    compare(mx.array([[55, 57], [59, 61]]))
    assert fast._batched is None
    compare(mx.array([[63], [65]]))
    assert fast._batched is not None


def test_decode_pack_preserves_snapshot_layout_and_is_conservative():
    def make():
        rows = [DeepseekV41Cache(2) for _ in range(2)]
        for i, row in enumerate(rows):
            row.offset = 7
            row.window[0] = mx.full((1, 4, 32), float(i))
            row.compress[1] = mx.full((1, 3, 32), float(i + 2))
            row.compress_kv = row.compress[1]
            row.engram = mx.full((1, 7), i, mx.int64)
        return BatchDeepseekV41Cache(rows)

    cache = make()
    before = cache.state
    packed = cache.for_decode(1)
    assert isinstance(packed, DeepseekV41Cache)
    assert cache.for_decode(1) is packed
    for original, snapshot in zip(before, cache.state):
        for a, b in zip(original, snapshot):
            assert mx.array_equal(a, b).item()
    assert cache._batched is packed
    assert cache.rows[1].offset == 7
    assert cache._batched is None

    for change in (
        lambda c: c.left_padding.__setitem__(0, 1),
        lambda c: c._lengths.__setitem__(0, 1),
        lambda c: setattr(c.rows[1], "offset", 8),
        lambda c: c.rows[1].window.__setitem__(0, None),
        lambda c: c.rows[1].window.__setitem__(0, mx.zeros((1, 5, 32))),
        lambda c: c.rows[1].window.__setitem__(0, mx.zeros((1, 4, 32), mx.bfloat16)),
    ):
        cache = make()
        change(cache)
        assert cache.for_decode(1) is cache
        assert cache._batched is None


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("owned", ["none", "some", "all"])
def test_device_routes_match_compact_owned_routes(dtype, owned):
    mx.random.seed(43)
    config = small_config()
    model = DeepseekV41MoE(config)
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    model.sharding_group = SimpleNamespace(rank=lambda: 1)
    count = model.switch_mlp.gate_proj.weight.shape[0]
    x = mx.random.normal((4, 1, config.hidden_size)).astype(dtype)
    if owned == "none":
        indices = mx.zeros((4, 1, 2), mx.int32)
    elif owned == "all":
        indices = mx.full((4, 1, 2), count, mx.int32)
    else:
        indices = mx.broadcast_to(mx.array([0, count + 1]), (4, 1, 2))
    scores = mx.random.uniform(shape=indices.shape)
    expected = []
    # Independent route-by-route oracle, including duplicate and zero-owned routes.
    for row in range(4):
        value = mx.zeros((1, 1, config.hidden_size), mx.float32)
        for route in range(2):
            index = indices[row, 0, route].item() - count
            if 0 <= index < count:
                y = model.switch_mlp(
                    x[row : row + 1],
                    mx.array([[[index]]]),
                    scores[row : row + 1, :, route : route + 1],
                )
                value = value + y[..., 0, :].astype(mx.float32)
        expected.append(value)
    with patch.object(mx.distributed, "all_sum", side_effect=lambda data, **_: data):
        actual = model._distributed_decode_experts(x, indices, scores)
    expected = mx.concatenate(expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
