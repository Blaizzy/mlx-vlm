import mlx.core as mx

from mlx_vlm.gliner import _CharSplitter, _resolve_flat_overlaps, _WhitespaceSplitter
from mlx_vlm.models.gliner2_5 import Model, ModelConfig
from mlx_vlm.models.gliner2_5.boundary import (
    DocumentCandidatePool,
    Marginals,
    PooledCandidates,
    SharedPoolScorer,
)


def test_nested_encoder_config():
    config = ModelConfig.from_dict(
        {
            "model_type": "extractor",
            "architecture": "boundary",
            "encoder_config": {
                "vocab_size": 250112,
                "hidden_size": 768,
                "num_attention_heads": 12,
            },
        }
    )

    assert config.model_type == "gliner2_5"
    assert config.encoder_config.vocab_size == 250112
    assert config.encoder_config.hidden_size == 768


def test_checkpoint_key_sanitization():
    weights = {
        "encoder.embeddings.LayerNorm.weight": mx.ones((4,)),
        "encoder.encoder.layer.0.attention.self.query_proj.weight": mx.ones((4, 4)),
        "boundary_head.boundary_proposer.end_key_projection.weight": mx.ones((4, 4)),
        "boundary_head.shared_pool_builder.start_projection.weight": mx.ones((4, 4)),
    }

    sanitized = Model.sanitize(None, weights)

    assert "encoder.embeddings.layer_norm.weight" in sanitized
    assert "encoder.encoder.layers.0.attention.self_attn.query_proj.weight" in sanitized
    assert "boundary_head.boundary_proposer.end_key_projection.weight" not in sanitized
    assert "boundary_head.shared_pool_builder.start_projection.weight" in sanitized


def test_word_splitters_preserve_offsets():
    text = "Email Me@Example.com 北京"

    whitespace = _WhitespaceSplitter()(text)
    characters = _CharSplitter()(text)

    assert whitespace[1] == ("me@example.com", 6, 20)
    assert characters[-2:] == [("北", 21, 22), ("京", 22, 23)]


def test_flat_overlap_resolution_uses_total_score():
    spans = [
        (0.8, 0, 2),
        (0.6, 0, 1),
        (0.6, 1, 2),
    ]

    assert _resolve_flat_overlaps(spans) == [(0.6, 0, 1), (0.6, 1, 2)]


def test_shared_pool_scorer_keeps_candidate_major_layout():
    """Span-marginal gathers must stay candidate-major.

    Indexing a ``(queries, positions)`` plane with a candidate vector puts the
    advanced axis first, so the gather is ``(candidates, queries)`` -- matching
    ``score``. Transposing it instead makes the scorer unusable for any input
    where the candidate and query counts differ.
    """
    batch, positions, candidates, queries = 1, 6, 5, 3
    hidden_size, boundary_dim = 16, 8
    scorer = SharedPoolScorer(
        hidden_size, boundary_dim, {"pair_dim": 8, "content_dim": 4}
    )

    starts = mx.array([[0, 1, 2, 0, 3]])
    ends = mx.array([[2, 3, 4, 5, 5]])
    pooled = PooledCandidates(
        indices=mx.stack((starts, ends), axis=-1),
        mask=mx.ones((batch, candidates), dtype=mx.bool_),
        compat_logits=mx.zeros((batch, candidates)),
    )
    marginals = Marginals(
        start_logits=mx.zeros((batch, queries, positions)),
        end_logits=mx.zeros((batch, queries, positions)),
        inside_prefix=mx.zeros((batch, queries, positions)),
        inside_mean=mx.zeros((batch, queries, 1)),
    )

    logits = scorer(
        mx.zeros((batch, positions, boundary_dim)),
        mx.zeros((batch, queries, hidden_size)),
        mx.ones((batch, queries), dtype=mx.bool_),
        pooled,
        marginals,
        mx.zeros((batch, positions, hidden_size)),
        mx.ones((batch, positions)),
    )

    assert logits.shape == (batch, candidates, queries)


def _zeroed_pool(boundary_dim, settings):
    pool = DocumentCandidatePool(boundary_dim, settings)
    # Zero the projections so pair compatibility drops out and the ranking is
    # exactly start_logits + end_logits.
    for projection in (pool.start_projection, pool.end_projection):
        projection.weight = mx.zeros_like(projection.weight)
        projection.bias = mx.zeros_like(projection.bias)
    return pool


def test_candidate_pool_promotes_each_query_best_pair():
    """Every active query must get its top-scoring pair into the shared pool.

    Uses 3 queries and 6 boundaries so the query axis and the pair axis have
    different lengths -- a gather that transposes them would pick the wrong
    pairs (or fail outright) instead of silently agreeing.
    """
    boundaries, queries = 6, 3
    settings = {"pool_boundary_top_k": 4, "pool_size": 6, "min_pool_per_query": 2}
    pool = _zeroed_pool(8, settings)

    start_logits = mx.zeros((1, queries, boundaries))
    end_logits = mx.zeros((1, queries, boundaries))
    # query 0 -> (0, 5), query 1 -> (1, 4), query 2 -> (2, 3)
    wanted = [(0, 5), (1, 4), (2, 3)]
    for query, (start, end) in enumerate(wanted):
        start_logits[0, query, start] = 10.0 + query
        end_logits[0, query, end] = 10.0 + query

    pooled = pool(
        mx.zeros((1, boundaries, 8)),
        mx.ones((1, boundaries), dtype=mx.bool_),
        mx.ones((1, queries), dtype=mx.bool_),
        start_logits,
        end_logits,
    )

    assert pooled.indices.shape == (1, settings["pool_size"], 2)
    assert pooled.mask.shape == (1, settings["pool_size"])
    selected = {
        (start, end)
        for (start, end), keep in zip(
            pooled.indices[0].tolist(), pooled.mask[0].tolist()
        )
        if keep
    }
    assert all(end > start for start, end in selected)
    for pair in wanted:
        assert pair in selected, f"{pair} missing from {sorted(selected)}"
