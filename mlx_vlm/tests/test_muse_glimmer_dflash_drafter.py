import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.muse_glimmer import Model as MuseGlimmerModel
from mlx_vlm.models.muse_glimmer import ModelConfig as MuseGlimmerConfig
from mlx_vlm.models.muse_glimmer import TextConfig, VisionConfig
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    Model as MuseGlimmerAssistantModel,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    ModelConfig,
    expected_muse_glimmer_assistant_weight_shapes,
    validate_muse_glimmer_assistant_weights,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant.dflash import (
    _bidirectional_sliding_mask,
    _prepare_assistant_mlp_input,
)
from mlx_vlm.utils import get_model_and_args


def _published_config():
    return {
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 6656,
        "intermediate_size": 19968,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 131072,
        "rope_parameters": {
            "rope_theta": 500000.0,
            "rope_type": "default",
        },
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 2048,
        "block_size": 16,
        "mask_token_id": 201818,
        "target_layer_ids": [1, 13, 25, 37, 49],
    }


def _tiny_assistant_config():
    return ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        block_size=4,
        mask_token_id=63,
        target_layer_ids=[0, 1],
        num_target_layers=2,
        vocab_size=64,
    )


def _tiny_target():
    text = TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    return MuseGlimmerModel(
        MuseGlimmerConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=7,
            video_token_id=6,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    )


def _generated_tokens(target, prompt, drafter=None):
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            prompt,
            target,
            None,
            None,
            max_tokens=10,
            temperature=0,
            prefill_step_size=None,
            draft_model=drafter,
            draft_kind="dflash",
        )
    ]


def test_published_config_and_weight_contract():
    config = ModelConfig.from_dict(_published_config())

    assert config.rope_theta == 500000.0
    assert config.target_layer_ids == [1, 13, 25, 37, 49]
    assert config.num_target_layers == 52
    assert config.vocab_size == 202048

    expected = expected_muse_glimmer_assistant_weight_shapes(config)
    assert len(expected) == 58
    assert expected["encoder.fc.weight"] == (6656, 33280)
    assert expected["layers.0.self_attn.o_proj.weight"] == (6656, 4096)
    assert expected["layers.4.mlp.down_proj.weight"] == (6656, 19968)

    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    validate_muse_glimmer_assistant_weights(weights, config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config.update({"layer_types": ["full_attention"] * 5}),
            "sliding_attention",
        ),
        (
            lambda config: config.update({"target_layer_ids": [1, 13, 25, 37, 52]}),
            "target_layer_ids",
        ),
        (
            lambda config: config.update({"mask_token_id": 202048}),
            "mask_token_id",
        ),
    ],
)
def test_invalid_checkpoint_contract_is_rejected(mutate, message):
    config = _published_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict(config)


def test_generic_loader_and_kind_detection(tmp_path):
    architecture, model_type = get_model_and_args(_published_config())
    assert model_type == "muse_glimmer_assistant"
    assert architecture.Model is MuseGlimmerAssistantModel

    (tmp_path / "config.json").write_text(json.dumps(_published_config()))
    assert resolve_drafter_kind(tmp_path, None) == "dflash"
    assert resolve_drafter_kind(tmp_path, "mtp") == "dflash"


def test_binding_uses_raw_target_embedding_and_checks_target_family():
    target = _tiny_target()
    drafter = MuseGlimmerAssistantModel(_tiny_assistant_config())

    validate_drafter_compatibility(target, drafter, "dflash")
    drafter.bind(target)
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    raw = target.language_model.model.embed_tokens(inputs)
    normalized = target.language_model.model.embed_norm(raw)
    actual = drafter._embed_input_tokens(inputs)
    mx.eval(raw, normalized, actual)

    assert bool(mx.array_equal(actual, raw).item())
    assert not bool(mx.array_equal(actual, normalized).item())

    target.language_model.config.model_type = "other"
    with pytest.raises(ValueError, match="Muse Glimmer text target"):
        validate_drafter_compatibility(target, drafter, "dflash")


def test_bidirectional_sliding_mask_matches_transformers_definition():
    mask = _bidirectional_sliding_mask(
        query_start=10,
        query_length=3,
        key_start=7,
        key_length=6,
        sliding_window=2,
    )
    assert mask.tolist() == [
        [False, True, True, True, True, True],
        [False, False, True, True, True, True],
        [False, False, False, True, True, True],
    ]


def test_compiled_assistant_transition_preserves_outputs():
    residual = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8).astype(mx.bfloat16)
    attention = (residual * 0.25 - 0.75).astype(mx.bfloat16)
    weight = (mx.arange(8, dtype=mx.float32) * 0.03 + 0.5).astype(mx.bfloat16)
    eps = 1e-5

    expected_hidden = residual + attention
    expected_mlp_input = mx.fast.rms_norm(expected_hidden, weight, eps)
    actual_hidden, actual_mlp_input = _prepare_assistant_mlp_input(
        residual, attention, weight, eps
    )
    mx.eval(
        expected_hidden,
        expected_mlp_input,
        actual_hidden,
        actual_mlp_input,
    )

    assert bool(mx.array_equal(actual_hidden, expected_hidden).item())
    assert bool(mx.array_equal(actual_mlp_input, expected_mlp_input).item())


def test_tiny_drafter_forward_and_cache_shapes():
    mx.random.seed(0)
    target = _tiny_target()
    drafter = MuseGlimmerAssistantModel(_tiny_assistant_config())
    caches = drafter.reset(target)
    target_output = target.language_model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        cache=target.make_cache(),
        capture_layer_ids=[0, 1],
    )
    hidden = mx.concatenate(target_output.hidden_states, axis=-1)
    tokens = drafter.draft_block(
        4,
        hidden,
        caches,
        block_size=4,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    mx.eval(tokens)

    assert hidden.shape == (1, 3, 32)
    assert tokens.shape == (1, 3)
    assert all(cache.offset == 3 for cache in caches)


def test_prepared_target_hidden_preserves_draft_tokens():
    mx.random.seed(4)
    target = _tiny_target()
    drafter = MuseGlimmerAssistantModel(_tiny_assistant_config())
    drafter.reset(target)
    target_output = target.language_model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        cache=target.make_cache(),
        capture_layer_ids=[0, 1],
    )
    hidden = mx.concatenate(target_output.hidden_states, axis=-1)
    prepared = drafter.prepare_target_hidden(hidden)
    sampler = lambda logits: mx.argmax(logits, axis=-1)

    regular_tokens = drafter.draft_block(
        4,
        hidden,
        drafter.make_cache(),
        block_size=4,
        sampler=sampler,
    )
    prepared_tokens = drafter.draft_block(
        4,
        prepared,
        drafter.make_cache(),
        block_size=4,
        sampler=sampler,
        target_hidden_prepared=True,
    )
    mx.eval(regular_tokens, prepared_tokens)

    assert bool(mx.array_equal(regular_tokens, prepared_tokens).item())


def test_greedy_speculative_generation_matches_baseline():
    mx.random.seed(7)
    target = _tiny_target()
    drafter = MuseGlimmerAssistantModel(_tiny_assistant_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    speculative = _generated_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens
