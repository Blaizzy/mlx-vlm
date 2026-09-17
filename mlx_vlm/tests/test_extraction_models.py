"""GLiNER span extraction, privacy tagging, checkpoint layouts, and quantized inference."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
from mlx_vlm.models.gliner2_5 import Model as GlinerModel
from mlx_vlm.models.gliner2_5 import ModelConfig as GlinerConfig
from mlx_vlm.models.gliner2_5.boundary import (
    DocumentCandidatePool,
    Marginals,
    PooledCandidates,
    SharedPoolScorer,
)
from mlx_vlm.models.openai_privacy_filter import Model as PrivacyModel
from mlx_vlm.models.openai_privacy_filter import ModelConfig as PrivacyConfig
from mlx_vlm.privacy_filter import PrivacyFilter
from mlx_vlm.utils import get_model_and_args, load_config


def test_checkpoint_key_sanitization():
    weights = {
        "encoder.embeddings.LayerNorm.weight": mx.ones((4,)),
        "encoder.encoder.layer.0.attention.self.query_proj.weight": mx.ones((4, 4)),
        "boundary_head.boundary_proposer.end_key_projection.weight": mx.ones((4, 4)),
        "boundary_head.shared_pool_builder.start_projection.weight": mx.ones((4, 4)),
    }

    sanitized = GlinerModel.sanitize(None, weights)

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


@pytest.mark.parametrize(
    "added,kwargs,error",
    [(0, {"word_splitter": "bpe"}, "word_splitter"), (4, {}, "special tokens")],
)
def test_gliner2_rejects_incompatible_tokenization(added, kwargs, error):
    tokenizer = SimpleNamespace(add_special_tokens=lambda _: added)
    with pytest.raises(ValueError, match=error):
        GLiNER2(object(), tokenizer, **kwargs)


def test_quantized_encoder_still_runs():
    """A quantized checkpoint must still encode.

    ``mlx_vlm.convert -q`` turns ``rel_embeddings`` into a QuantizedEmbedding,
    whose ``.weight`` is packed uint32 with a narrower last dimension. Reading
    that attribute directly instead of calling the module makes the relative
    embedding LayerNorm reject the shape, so every quantized model fails at
    inference even though conversion reports success.
    """
    config = GlinerConfig.from_dict(
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
    model = GlinerModel(config)
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


@pytest.mark.parametrize(
    "sidecar,inline",
    [(True, False), (False, True), (False, False)],
    ids=["sidecar", "inline", "missing"],
)
def test_gliner_encoder_config_loading(tmp_path, sidecar, inline):
    root = _write_checkpoint(tmp_path, sidecar=sidecar, inline=inline)
    if sidecar or inline:
        assert load_config(root)["encoder_config"]["hidden_size"] == 64
    else:
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


LABELS = {0: "O", 1: "B-x", 2: "I-x", 3: "E-x", 4: "S-x"}


def _privacy_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "sliding_window": 2,
        "max_position_embeddings": 32,
        "default_n_ctx": 32,
        "pad_token_id": 31,
        "eos_token_id": 31,
        "num_labels": 5,
        "id2label": LABELS,
        "attention_chunk_size": 3,
        "moe_chunk_size": 3,
    }
    values.update(overrides)
    return PrivacyConfig(**values)


def test_checkpoint_expert_layout_sanitization():
    gate_up = mx.arange(2 * 3 * 8).reshape(2, 3, 8)
    down = mx.arange(2 * 4 * 3).reshape(2, 4, 3)
    weights = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((2, 8)),
        "model.layers.0.mlp.experts.down_proj": down,
        "model.layers.0.mlp.experts.down_proj_bias": mx.zeros((2, 3)),
        "score.weight": mx.zeros((5, 3)),
    }

    sanitized = PrivacyModel.sanitize(None, weights)

    gate_key = "model.layers.0.mlp.experts.gate_proj.weight"
    up_key = "model.layers.0.mlp.experts.up_proj.weight"
    down_key = "model.layers.0.mlp.experts.down_proj.weight"
    transposed = gate_up.swapaxes(-1, -2)
    assert sanitized[gate_key].shape == (2, 4, 3)
    assert sanitized[up_key].shape == (2, 4, 3)
    assert sanitized[down_key].shape == (2, 3, 4)
    assert mx.array_equal(sanitized[gate_key], transposed[:, :4]).item()
    assert mx.array_equal(sanitized[up_key], transposed[:, 4:]).item()
    assert mx.array_equal(sanitized[down_key], down.swapaxes(-1, -2)).item()
    assert "model.layers.0.mlp.experts.gate_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.up_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.down_proj.bias" in sanitized
    assert "score.weight" in sanitized


@pytest.mark.parametrize(
    "quantization",
    [
        None,
        (64, 4, "affine"),
        (64, 5, "affine"),
        (64, 6, "affine"),
        (64, 8, "affine"),
        (32, 4, "mxfp4"),
        (16, 4, "nvfp4"),
        (32, 8, "mxfp8"),
    ],
    ids=[
        "float-chunk-boundaries",
        "affine4",
        "affine5",
        "affine6",
        "affine8",
        "mxfp4",
        "nvfp4",
        "mxfp8",
    ],
)
def test_privacy_forward(quantization):
    overrides = (
        dict(hidden_size=64, intermediate_size=64, head_dim=16, num_attention_heads=4)
        if quantization
        else {}
    )
    model = PrivacyModel(_privacy_config(**overrides))
    if quantization:
        group_size, bits, mode = quantization
        nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(input_ids, attention_mask=mx.ones_like(input_ids)).logits
    mx.eval(logits)
    assert logits.shape == (1, 5, 5)
    assert mx.all(mx.isfinite(logits)).item()


class _PrivacyTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "Alice emailed bob@example.com"
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 5), (5, 13), (13, 29)]}


class _PrivacyModel:
    def __init__(self):
        labels = {
            0: "O",
            1: "B-private_person",
            2: "I-private_person",
            3: "E-private_person",
            4: "S-private_person",
            5: "B-private_email",
            6: "I-private_email",
            7: "E-private_email",
            8: "S-private_email",
        }
        self.config = SimpleNamespace(
            id2label=labels, num_labels=len(labels), default_n_ctx=16
        )

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        logits = mx.full((1, input_ids.shape[1], 9), -10.0)
        logits[0, 0, 4] = 10.0
        logits[0, 1, 0] = 10.0
        logits[0, 2, 8] = 10.0
        return SimpleNamespace(logits=logits)


def test_high_level_api_returns_offsets_and_redacted_text():
    detector = PrivacyFilter(_PrivacyModel(), _PrivacyTokenizer())

    result = detector("Alice emailed bob@example.com")

    assert [(span.label, span.start, span.end) for span in result.spans] == [
        ("private_person", 0, 5),
        ("private_email", 14, 29),
    ]
    assert result.redacted_text == ("<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>")
    assert result.to_dict()["spans"][0]["text"] == "Alice"
