"""Single-request decoding must release cache graphs unused by the logits."""

import io

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar
from mlx_vlm.models import deepseek_v4
from mlx_vlm.models.cache import CacheList


def _graph_edges(array):
    graph = io.StringIO()
    mx.export_to_dot(graph, array)
    return graph.getvalue().count("->")


@pytest.fixture
def model():
    mx.random.seed(0)
    config = deepseek_v4.ModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=4,
        sliding_window=16,
        compress_ratios=[0, 128, 4],
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    model = deepseek_v4.Model(config)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("prompt_length", [1, 65])
def test_decode_bounds_unused_cache_graphs_without_changing_outputs(
    model, prompt_length, monkeypatch
):
    def decode():
        cache = model.make_cache()
        local_caches = [c[0] if isinstance(c, CacheList) else c for c in cache]
        graph_sizes, tokens, logprobs = [], [], []
        generator = ar.generate_step(
            mx.arange(1, prompt_length + 1)[None],
            model,
            pixel_values=None,
            mask=None,
            prompt_cache=cache,
            prefill_step_size=32,
            max_tokens=125,
            temperature=0,
        )
        try:
            for token, lp in generator:
                tokens.append(token)
                logprobs.append(lp)
                graph_sizes.append([_graph_edges(c.values) for c in local_caches])
            result = mx.stack(logprobs)
            mx.eval(result)
            return tokens, result, graph_sizes
        finally:
            generator.close()
            mx.eval([c.state for c in cache])

    tokens, logprobs, graph_sizes = decode()
    assert len(tokens) == 125
    assert max(max(sizes) for sizes in graph_sizes) < 400
    assert graph_sizes[49] == [0, 0, 0]
    assert graph_sizes[99] == [0, 0, 0]

    monkeypatch.setattr(ar, "DEFAULT_CACHE_EVAL_INTERVAL", 10**9)
    reference_tokens, reference_logprobs, unbounded = decode()
    assert min(unbounded[-1]) > 800
    assert tokens == reference_tokens
    assert mx.array_equal(logprobs, reference_logprobs).item()
