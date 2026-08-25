import mlx.core as mx

from mlx_vlm.gliner import _CharSplitter, _resolve_flat_overlaps, _WhitespaceSplitter
from mlx_vlm.models.gliner2_5 import Model, ModelConfig


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
