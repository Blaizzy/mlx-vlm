import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.models.embedding_gemma2 import Model, ModelConfig, TextConfig


def tiny_config(**kwargs):
    return ModelConfig(
        text_config=TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            hidden_size_per_layer_input=8,
            embedding_dim=24,
            sliding_window=3,
            layer_types=["sliding_attention", "full_attention"],
            per_layer_config={"01": {"head_dim": 16}},
        ),
        image_token_id=60,
        audio_token_id=61,
        video_token_id=62,
        **kwargs,
    )


@pytest.mark.parametrize("left_padding", [False, True])
def test_padded_batch_matches_individual(left_padding):
    mx.random.seed(42)
    model = Model(tiny_config())
    ids = [1, 2, 3]
    solo = model(mx.array([ids])).text_embeds
    padded = [0, 0, *ids] if left_padding else [*ids, 0, 0]
    valid = [0, 0, 1, 1, 1] if left_padding else [1, 1, 1, 0, 0]
    output = model(
        mx.array([padded, [4, 5, 6, 7, 8]]),
        attention_mask=mx.array([valid, [1, 1, 1, 1, 1]]),
    )
    np.testing.assert_allclose(output.text_embeds[:1], solo, atol=1e-5, rtol=1e-5)


def test_attention_is_bidirectional():
    model = Model(tiny_config())
    first = model(mx.array([[1, 2, 3]])).last_hidden_state
    second = model(mx.array([[1, 2, 4]])).last_hidden_state
    assert not mx.allclose(first[:, 0], second[:, 0])


def test_sliding_attention_includes_window_boundary():
    mx.random.seed(42)
    model = Model(tiny_config())
    # Isolate the first (sliding) layer so the full layer cannot mix distant tokens.
    model.language_model.layers = model.language_model.layers[:1]
    baseline = model(mx.array([[1, 2, 3, 4, 5]])).last_hidden_state[:, 0]
    boundary = model(mx.array([[1, 2, 3, 6, 5]])).last_hidden_state[:, 0]
    outside = model(mx.array([[1, 2, 3, 4, 6]])).last_hidden_state[:, 0]
    assert not mx.allclose(baseline, boundary)
    np.testing.assert_allclose(baseline, outside, atol=1e-6)


def test_explicit_position_ids():
    mx.random.seed(42)
    model = Model(tiny_config())
    ids = mx.array([[1, 2, 3]])
    default = model(ids).last_hidden_state
    explicit = model(ids, position_ids=mx.array([[0, 1, 2]])).last_hidden_state
    spaced = model(ids, position_ids=mx.array([[0, 2, 4]])).last_hidden_state
    np.testing.assert_array_equal(default, explicit)
    assert not mx.allclose(default, spaced)


def test_full_attention_config_defaults_and_index_normalization():
    config = TextConfig(
        num_hidden_layers=7,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        hidden_size_per_layer_input=8,
    )
    model = Model(ModelConfig(text_config=config))
    assert config.layer_types[-1] == "full_attention"
    assert model.layers[5].self_attn.head_dim == 512
    assert model.layers[5].self_attn.num_kv_heads == 1
    config = tiny_config()
    config.text_config.per_layer_config = {1: {"head_dim": 16}}
    config.text_config.__post_init__()
    assert Model(config).layers[1].self_attn.head_dim == 16


def test_media_scatter_preserves_batch_order_and_validates_counts():
    ids = mx.array([[1, 60, 60], [60, 2, 0]])
    embeddings = mx.zeros((2, 3, 4))
    features = mx.arange(12).reshape(3, 4)
    output = Model._scatter(embeddings, ids, 60, features)
    np.testing.assert_array_equal(output[0, 1:], features[:2])
    np.testing.assert_array_equal(output[1, 0], features[2])
    np.testing.assert_array_equal(output[1, 1:], mx.zeros((2, 4)))
    with pytest.raises(ValueError, match="token count"):
        Model._scatter(embeddings, ids, 60, features[:2])


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_disabled_towers_reject_media(modality):
    model = Model(tiny_config())
    with pytest.raises(ValueError, match="require.*config"):
        getattr(model, f"get_{modality}_features")(mx.zeros((1, 2, 3)), None)


def test_sanitize_and_converted_checkpoint_roundtrip(tmp_path):
    from dataclasses import asdict

    from mlx_vlm.models.embedding_gemma2 import AudioConfig

    config = tiny_config(
        audio_config=AudioConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            subsampling_conv_channels=(4, 2),
            output_proj_dims=8,
        )
    )
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    original = {}
    for key, value in weights.items():
        if key.endswith("conv.weight"):
            value = value.transpose(0, 3, 1, 2)
        elif key.endswith("depthwise_conv1d.weight"):
            value = value.transpose(0, 2, 1)
        original["model." + key] = value
    sanitized = model.sanitize(original)
    twice = model.sanitize(sanitized)
    assert weights.keys() == sanitized.keys() == twice.keys()
    for key in weights:
        np.testing.assert_array_equal(sanitized[key], weights[key])
        np.testing.assert_array_equal(twice[key], weights[key])

    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    loaded = load_embedding_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_allclose(
        loaded(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )

    config.audio_config = None
    reduced = load_embedding_model(tmp_path, config=asdict(config))
    assert reduced.audio_tower is None
    np.testing.assert_allclose(
        reduced(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )
