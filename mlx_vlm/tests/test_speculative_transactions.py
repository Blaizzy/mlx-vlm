from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.deepseek_v4 import language as deepseek_language
from mlx_vlm.models.qwen4_exp.language import LanguageModel as QwenLanguageModel
from mlx_vlm.speculative.cache_state import CacheTransaction
from mlx_vlm.tests.test_qwen4_decode import _outer_config, _tiny_text_config
from mlx_vlm.tests.test_speculative import _tiny_glm5_next_text_config


def _tiny_deepseek_v4_config():
    return deepseek_language.ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        head_dim=8,
        o_groups=1,
        o_lora_rank=8,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=1,
        num_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        compress_ratios=[0],
        sliding_window=16,
        max_position_embeddings=128,
    )


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("state_steps", [1, 3])
def test_qwen_recurrent_history_uses_allocated_batch_stride(batch, state_steps):
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update_with_states

    mx.random.seed(2127)
    length, heads, width = 4, 2, 32
    shape = (batch, length, heads, width)
    q, k, v = [mx.random.normal(shape).astype(mx.bfloat16) * 0.1 for _ in range(3)]
    a, b = [mx.random.normal(shape[:-1]).astype(mx.bfloat16) for _ in range(2)]
    args = (q, k, v, a, b, mx.zeros((heads,)), mx.zeros((heads,)))
    output, state, history = gated_delta_update_with_states(
        *args, state_steps=state_steps
    )
    mx.eval(output, state, history)
    # Each independently executed row is an oracle with no batch stride.
    for row in range(batch):
        expected = gated_delta_update_with_states(
            *(x[row : row + 1] for x in args[:5]), *args[5:], state_steps=state_steps
        )
        for actual, reference in zip((output, state, history), expected):
            assert mx.array_equal(actual[row : row + 1], reference).item()


def test_glm_prefill_indexer_projections_are_chunk_invariant():
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hidden_size = 4096
    config.q_lora_rank = 1536
    config.index_head_dim = 128
    config.index_n_heads = 32
    indexer = Glm5NextIndexer(config, 0)
    indexer.set_dtype(mx.bfloat16)
    indexer.index_kpool_compress_gate = mx.random.normal((128, 4096)) * 0.01
    nn.quantize(indexer, group_size=64, bits=4)
    x = mx.random.normal((1, 4096, 4096)).astype(mx.bfloat16)
    q = mx.random.normal((1, 4096, 1536)).astype(mx.bfloat16)
    reference = (*indexer._project_keys(x), *indexer._project_queries(x, q))
    mx.eval(reference)
    parts = [
        (
            *indexer._project_keys(x[:, start : start + 2048]),
            *indexer._project_queries(
                x[:, start : start + 2048], q[:, start : start + 2048]
            ),
        )
        for start in (0, 2048)
    ]
    for expected, segments in zip(reference, zip(*parts)):
        assert mx.array_equal(expected, mx.concatenate(segments, axis=1)).item()


def test_hyperconnection_prefill_is_chunk_invariant():
    from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection

    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hidden_size, config.hc_mult = 4096, 4
    connection = HyperConnection(config)
    connection.fn = mx.random.normal(connection.fn.shape) * 0.01
    connection.eval()
    x = mx.random.normal((1, 4096, 4, 4096)).astype(mx.bfloat16)
    expected = connection(x)
    mx.eval(expected)
    parts = [connection(x[:, :2048]), connection(x[:, 2048:])]
    for reference, segments in zip(expected, zip(*parts)):
        assert mx.array_equal(reference, mx.concatenate(segments, axis=1)).item()


def test_ordinary_qwen4_decode_does_not_record_speculation():
    config = _tiny_text_config()
    config.linear_key_head_dim = config.linear_value_head_dim = 32
    model = QwenLanguageModel(config, _outer_config())
    model.eval()
    caches = model.make_cache()
    mx.eval(model(mx.array([[1, 2]]), cache=caches).logits)
    for token in (3, 4):
        output = model(mx.array([[token]]), cache=caches)
        mx.eval(output.logits)
        assert not hasattr(output, "gdn_states")
        for cache in caches:
            if isinstance(cache, ArraysCache):
                assert not cache.is_speculating
                assert cache.nbytes == sum(
                    x.nbytes for x in cache.state if x is not None
                )


@pytest.mark.parametrize("family", ["glm", "qwen"])
@pytest.mark.parametrize("step", [1, 2, 5])
def test_ordinary_linear_attention_uses_temporal_cache_without_adapter(family, step):
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    mx.random.seed(2127)
    if family == "glm":
        config = _tiny_glm5_next_text_config()
        config.linear_head_dim = 32
        layer = Glm5NextLinearAttention(config)
    else:
        config = _tiny_text_config()
        config.linear_key_head_dim = config.linear_value_head_dim = 32
        layer = Qwen3_5GatedDeltaNet(config)
    layer.eval()
    cache = ArraysCache(2, left_padding=[0] * 3)
    cache.prepare(lengths=[100] * 3)
    prefix = mx.random.normal((3, 4, config.hidden_size))
    mx.eval(layer(prefix, cache=cache), cache.state)
    assert cache.history_capacity == 0

    for retained in ([0, 2, 5], [3, 1, 4]):
        inputs = mx.random.normal((3, 5, config.hidden_size))
        reference = deepcopy(cache)
        states = [list(reference.state)]
        for index in range(5):
            mx.eval(layer(inputs[:, index : index + 1], cache=reference))
            states.append(list(reference.state))
        initial_lengths = cache.lengths
        transaction = CacheTransaction([cache], 5)
        for index in range(0, 5, step):
            mx.eval(layer(inputs[:, index : index + step], cache=cache))
        transaction.commit(retained)
        for slot in range(2):
            expected = mx.concatenate(
                [states[keep][slot][row : row + 1] for row, keep in enumerate(retained)]
            )
            assert mx.allclose(cache[slot], expected, rtol=0, atol=1e-6).item()
        assert mx.array_equal(
            cache.lengths, initial_lengths - mx.array(retained)
        ).item()
        assert cache.history_capacity == 0
        assert cache.nbytes == sum(value.nbytes for value in cache.state)


@pytest.mark.parametrize("family", ["glm", "deepseek"])
def test_shared_moe_preserves_expert_gradients_and_unweighted_replacements(family):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4MoE
    from mlx_vlm.models.glm5_next.language import Glm5NextMoE
    from mlx_vlm.models.switch_layers import SwitchGLU

    if family == "glm":
        config = _tiny_glm5_next_text_config()
        module = Glm5NextMoE(config)
        kwargs = {}
    else:
        config = _tiny_deepseek_v4_config()
        module = DeepseekV4MoE(config, 0)
        kwargs = dict(input_ids=mx.array([[1, 2, 3]]))
    inputs = mx.random.normal((1, 3, config.hidden_size))
    module.gate.freeze()
    value, grad = nn.value_and_grad(module, lambda m: m(inputs, **kwargs).sum())(module)
    mx.eval(value, grad)
    assert mx.isfinite(value).item()
    module.eval()
    original = module.switch_mlp
    expected = module(inputs, **kwargs)

    class ExternalExperts(nn.Module):
        def __call__(self, x, indices):
            return original(x, indices)

    module.switch_mlp = ExternalExperts()
    actual = module(inputs, **kwargs)
    assert mx.allclose(actual, expected, atol=1e-5).item()
    assert not isinstance(module.switch_mlp, SwitchGLU)


def test_transaction_scope_aborts_uncommitted_temporal_and_kv_updates():
    recurrent = ArraysCache(1)
    recurrent[0] = mx.zeros((1, 1))
    kv = KVCache()
    initial = mx.zeros((1, 1, 2, 1))
    kv.update_and_fetch(initial, initial)
    with pytest.raises(RuntimeError, match="injected failure"):
        with CacheTransaction([recurrent, kv], 2):
            recurrent[0] = mx.ones((1, 1))
            kv.update_and_fetch(initial, initial)
            raise RuntimeError("injected failure")
    assert recurrent[0].item() == 0
    assert not recurrent.is_speculating
    assert kv.offset == 2
