import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.nemotron_h_nano_omni.audio import (
    SoundEncoder,
    SoundFeatureExtractor,
    SoundProjection,
    sanitize_audio_weights,
)
from mlx_vlm.models.nemotron_h_nano_omni.config import (
    ModelConfig,
    SoundConfig,
    TextConfig,
    VisionConfig,
)
from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import Model


def tiny_text_config(hidden_size=24):
    return TextConfig(
        model_type="nemotron_h",
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_bias=False,
        mamba_num_heads=1,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=3,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=False,
        hybrid_override_pattern=["*"],
    )


def tiny_vision_config():
    return VisionConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        image_size=32,
        patch_size=16,
        max_resolution=32,
        args={"teachers": [{"name": "test"}]},
    )


def tiny_sound_config(hidden_size=16):
    return SoundConfig(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        projection_hidden_size=32,
    )


def test_sound_feature_extractor_shapes_and_masks():
    pytest.importorskip("mlx_audio")
    config = SoundConfig()
    extractor = SoundFeatureExtractor(config)
    waveform = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    features, mask, lengths = extractor([waveform])

    assert features.shape == (1, 11, config.num_mel_bins)
    assert mask.shape == (1, 11)
    assert lengths.tolist() == [11]
    assert mask.sum(axis=1).tolist() == [10]
    assert np.isfinite(np.array(features)).all()


def test_sound_encoder_and_projection_tiny_smoke():
    config = tiny_sound_config()
    encoder = SoundEncoder(config)
    projection = SoundProjection(config, llm_hidden_size=24)
    encoder.eval()
    projection.eval()

    features = mx.random.normal((1, 17, config.num_mel_bins))
    mask = mx.ones((1, 17), dtype=mx.int32)
    mask[:, -1] = 0

    encoded = encoder(features, mask)
    projected = projection(encoded)

    assert encoded.shape == (1, 3, config.hidden_size)
    assert projected.shape == (1, 3, 24)
    assert np.isfinite(np.array(encoded)).all()
    assert np.isfinite(np.array(projected)).all()


def test_model_merges_sound_features_into_input_embeddings():
    model = Model(
        ModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=tiny_sound_config(),
            projector_hidden_size=32,
            vit_hidden_size=16,
            img_context_token_id=98,
            sound_context_token_id=99,
        )
    )
    model.eval()

    input_ids = mx.array([[1, 99, 99, 99, 2]])
    input_features = mx.random.normal((1, 17, 16))
    feature_attention_mask = mx.ones((1, 17), dtype=mx.int32)
    base_embeddings = model.language_model.get_input_embeddings(input_ids)

    output = model.get_input_embeddings(
        input_ids,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    assert output.inputs_embeds.shape == base_embeddings.shape
    assert np.allclose(
        np.array(output.inputs_embeds[:, [0, 4], :]),
        np.array(base_embeddings[:, [0, 4], :]),
    )
    assert not np.allclose(
        np.array(output.inputs_embeds[:, 1:4, :]),
        np.array(base_embeddings[:, 1:4, :]),
    )


def test_model_rejects_sound_token_feature_count_mismatch():
    model = Model(
        ModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=tiny_sound_config(),
            projector_hidden_size=32,
            vit_hidden_size=16,
            img_context_token_id=98,
            sound_context_token_id=99,
        )
    )
    model.eval()

    with pytest.raises(ValueError, match="Sound token count"):
        model.get_input_embeddings(
            mx.array([[1, 99, 99, 2]]),
            input_features=mx.random.normal((1, 17, 16)),
            feature_attention_mask=mx.ones((1, 17), dtype=mx.int32),
        )


def test_sanitize_audio_and_projection_weights():
    weights = {
        "sound_encoder.encoder.feature_extractor.window": mx.ones((2,)),
        "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked": mx.array(0),
        "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight": mx.zeros(
            (8, 4, 3)
        ),
        "sound_encoder.encoder.subsampling.layers.0.weight": mx.zeros((4, 1, 3, 3)),
        "mlp1.0.weight": mx.ones((4,)),
        "mlp1.1.weight": mx.ones((4, 4)),
        "mlp1.3.weight": mx.ones((4, 4)),
    }

    audio_sanitized = sanitize_audio_weights(weights)
    assert "sound_encoder.encoder.feature_extractor.window" not in audio_sanitized
    assert (
        "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked"
        not in audio_sanitized
    )
    assert audio_sanitized[
        "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight"
    ].shape == (8, 3, 4)
    assert audio_sanitized[
        "sound_encoder.encoder.subsampling.layers.0.weight"
    ].shape == (
        4,
        3,
        3,
        1,
    )

    model = Model(
        ModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=None,
            projector_hidden_size=4,
            vit_hidden_size=1,
            img_context_token_id=98,
        )
    )
    model_sanitized = model.sanitize(weights)
    assert "mlp1.layers.0.weight" in model_sanitized
    assert "mlp1.layers.1.weight" in model_sanitized
    assert "mlp1.layers.3.weight" in model_sanitized
