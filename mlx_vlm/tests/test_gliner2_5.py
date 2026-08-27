import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.gliner import (
    GLiNER2,
    _CharSplitter,
    _resolve_flat_overlaps,
    _schema_tokens,
    _WhitespaceSplitter,
)
from mlx_vlm.models.gliner2_5 import Model, ModelConfig
from mlx_vlm.models.gliner2_5.boundary import (
    DocumentCandidatePool,
    Marginals,
    PooledCandidates,
    SharedPoolScorer,
)
from mlx_vlm.utils import get_model_and_args, load_config


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


def test_schema_tokens_place_markers_where_prepare_looks_for_them():
    """``_prepare`` derives query positions from fixed slots in the schema.

    It treats index 1 as the prompt marker and every second index from 4
    onward as a label marker, so the schema layout and that arithmetic have to
    agree or the query states are read from the wrong tokens.
    """
    labels = ["person", "company", "location"]
    schema = _schema_tokens("entities", labels, "[E]")

    assert schema[0] == "("
    assert schema[1] == "[P]"
    assert schema[2] == "entities"
    assert schema[3] == "("
    assert schema[-2:] == [")", ")"]

    marker_slots = {1, *range(4, len(schema) - 2, 2)}
    assert len(marker_slots) == len(labels) + 1
    for slot in sorted(marker_slots)[1:]:
        assert schema[slot] == "[E]"
        assert schema[slot + 1] in labels


def test_schema_tokens_carry_prompt_and_descriptions():
    schema = _schema_tokens(
        "sentiment",
        ["positive", "negative"],
        "[L]",
        prompt="rate the review",
        descriptions={"positive": "approving tone"},
    )

    prompt_text = schema[2]
    assert prompt_text.startswith("sentiment: rate the review")
    assert "[DESCRIPTION] positive: approving tone" in prompt_text
    # Undescribed labels must not gain a description clause.
    assert "negative:" not in prompt_text


def test_flat_overlap_resolution_keeps_disjoint_spans():
    spans = [(0.9, 0, 2), (0.8, 3, 5), (0.7, 6, 7)]

    assert _resolve_flat_overlaps(spans) == spans


def test_flat_overlap_resolution_drops_lower_scoring_overlap():
    spans = [(0.9, 0, 3), (0.4, 2, 4)]

    assert _resolve_flat_overlaps(spans) == [(0.9, 0, 3)]


def test_char_splitter_keeps_latin_runs_but_splits_cjk():
    text = "iPhone 15 在北京"

    pieces = _CharSplitter()(text)

    assert pieces[0] == ("iphone", 0, 6)
    assert pieces[1] == ("15", 7, 9)
    assert [piece[0] for piece in pieces[2:]] == ["在", "北", "京"]
    for token, start, end in pieces:
        assert text[start:end].lower() == token


def test_whitespace_splitter_offsets_round_trip():
    text = "Visit https://example.com or mail a.b@c.io — thanks!"

    for token, start, end in _WhitespaceSplitter()(text):
        assert text[start:end].lower() == token


def test_gliner2_rejects_unknown_word_splitter():
    class _Tokenizer:
        def add_special_tokens(self, _):
            return 0

    with pytest.raises(ValueError, match="word_splitter"):
        GLiNER2(object(), _Tokenizer(), word_splitter="bpe")


def test_gliner2_rejects_tokenizer_missing_special_tokens():
    class _Tokenizer:
        def add_special_tokens(self, _):
            return 4  # pretends it had to add them

    with pytest.raises(ValueError, match="special tokens"):
        GLiNER2(object(), _Tokenizer())


def test_quantized_encoder_still_runs():
    """A quantized checkpoint must still encode.

    ``mlx_vlm.convert -q`` turns ``rel_embeddings`` into a QuantizedEmbedding,
    whose ``.weight`` is packed uint32 with a narrower last dimension. Reading
    that attribute directly instead of calling the module makes the relative
    embedding LayerNorm reject the shape, so every quantized model fails at
    inference even though conversion reports success.
    """
    config = ModelConfig.from_dict(
        {
            "model_type": "extractor",
            "architecture": "boundary",
            "encoder_config": {
                "vocab_size": 512,
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "position_buckets": 64,
                "max_relative_positions": 128,
            },
        }
    )
    model = Model(config)
    model.eval()
    ids = mx.zeros((1, 16), dtype=mx.int32)
    mask = mx.ones((1, 16), dtype=mx.bool_)
    reference = model.encode(ids, mask)
    mx.eval(reference)

    nn.quantize(model.encoder, group_size=64, bits=4)
    mx.eval(model.parameters())

    assert isinstance(
        model.encoder.encoder.rel_embeddings, nn.QuantizedEmbedding
    ), "rel_embeddings should have been quantized for this test to be meaningful"
    encoded = model.encode(ids, mask)
    mx.eval(encoded)

    assert encoded.shape == reference.shape
    assert bool(mx.all(mx.isfinite(encoded.astype(mx.float32))))


def _write_checkpoint(root, *, sidecar=True, inline=False):
    encoder = {"vocab_size": 128, "hidden_size": 64, "num_attention_heads": 4}
    config = {
        "model_type": "extractor",
        "architecture": "boundary",
        "architectures": ["BoundaryExtractor"],
    }
    if inline:
        config["encoder_config"] = encoder
    (root / "config.json").write_text(json.dumps(config))
    if sidecar:
        (root / "encoder_config").mkdir()
        (root / "encoder_config" / "config.json").write_text(json.dumps(encoder))
    return root


def test_load_config_folds_in_the_sidecar_encoder_config(tmp_path):
    """The encoder config has to be composed in ``load_config``, not ``load_model``.

    Several callers -- ``fetch_from_hub``, ``load_image_processor``, the encoder
    and reranker loaders -- use ``load_config`` without ever reaching
    ``load_model``, so folding it in later hands them an incomplete config.
    """
    config = load_config(_write_checkpoint(tmp_path))

    assert config["encoder_config"]["hidden_size"] == 64


def test_load_config_accepts_an_inlined_encoder_config(tmp_path):
    """A converted checkpoint may carry the encoder config inline.

    ``convert`` only reproduces the sidecar directory because it copies every
    subdirectory wholesale; requiring the directory would couple loading to that
    incidental behaviour.
    """
    config = load_config(_write_checkpoint(tmp_path, sidecar=False, inline=True))

    assert config["encoder_config"]["hidden_size"] == 64


def test_load_config_reports_the_missing_encoder_config(tmp_path):
    """The error must name the sidecar, not read as a missing config.json."""
    root = _write_checkpoint(tmp_path, sidecar=False)

    with pytest.raises(FileNotFoundError, match="encoder config not found"):
        load_config(root)


def test_boundary_architecture_selects_the_gliner_module():
    _, model_type = get_model_and_args(
        {
            "model_type": "extractor",
            "architecture": "boundary",
            "architectures": ["BoundaryExtractor"],
        }
    )

    assert model_type == "gliner2_5"
