"""Single-request decoding must release cache graphs unused by the logits."""

import io

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import CacheList, RotatingKVCache


def _graph_edges(array):
    graph = io.StringIO()
    mx.export_to_dot(graph, array)
    return graph.getvalue().count("->")


class KeyOnlyModel:
    def __init__(self):
        self.language_model = self

    def get_input_embeddings(self, inputs, *args, **kwargs):
        return InputEmbeddingsFeatures(inputs_embeds=inputs[..., None])

    def __call__(self, inputs, cache, **kwargs):
        entry = cache[0][0] if isinstance(cache[0], CacheList) else cache[0]
        keys = inputs.astype(mx.float32)[:, None, :, None]
        keys, _ = entry.update_and_fetch(keys, mx.zeros((*keys.shape[:-1], 0)))
        logits = -((mx.arange(4) - keys.sum() % 4) ** 2)
        return LanguageModelOutput(logits=logits[None, None])


@pytest.mark.parametrize("prompt_length", [1, 65])
@pytest.mark.parametrize("nested", [False, True])
def test_decode_bounds_unused_cache_graphs_without_changing_outputs(
    prompt_length, nested, monkeypatch
):
    def decode():
        cache = RotatingKVCache(max_size=16)
        prompt_cache = [CacheList(cache) if nested else cache]
        graph_sizes, tokens, logprobs = [], [], []
        generator = ar.generate_step(
            mx.arange(1, prompt_length + 1)[None],
            KeyOnlyModel(),
            pixel_values=None,
            mask=None,
            prompt_cache=prompt_cache,
            prefill_step_size=32,
            max_tokens=125,
            temperature=0,
        )
        try:
            for token, lp in generator:
                tokens.append(token)
                logprobs.append(lp)
                graph_sizes.append(_graph_edges(cache.values))
            return tokens, mx.stack(logprobs), graph_sizes
        finally:
            generator.close()
            mx.eval(cache.state)

    tokens, logprobs, graph_sizes = decode()
    assert len(tokens) == 125
    assert graph_sizes[49] == graph_sizes[99] == 0
    assert max(graph_sizes[50:]) <= max(graph_sizes[:50])

    monkeypatch.setattr(ar, "DEFAULT_CACHE_EVAL_INTERVAL", len(tokens) + 1)
    reference_tokens, reference_logprobs, unbounded = decode()
    assert unbounded[-1] > max(graph_sizes)
    assert tokens == reference_tokens
    assert mx.array_equal(logprobs, reference_logprobs).item()
