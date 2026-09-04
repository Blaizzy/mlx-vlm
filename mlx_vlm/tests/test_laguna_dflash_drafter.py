from types import SimpleNamespace

import pytest

from mlx_vlm.speculative.drafters import validate_drafter_compatibility
from mlx_vlm.speculative.drafters.laguna_dflash import ModelConfig
from mlx_vlm.speculative.drafters.laguna_dflash.config import (
    expected_laguna_dflash_weight_shapes,
    validate_laguna_dflash_target,
    validate_laguna_dflash_weights,
)


def _config_dict():
    return {
        "model_type": "laguna",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 6,
        "num_attention_heads": 72,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1048576,
        "rope_theta": 500000.0,
        "vocab_size": 100352,
        "draft_vocab_size": 100352,
        "layer_types": ["sliding_attention"] * 6,
        "sliding_windows": [512] * 6,
        "sliding_window": 512,
        "gating": "per-head",
        "eagle_aux_hidden_state_layer_ids": [2, 11, 20, 30, 39, 48],
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 48,
            "target_layer_ids": [1, 10, 19, 29, 38, 47],
            "causal": True,
        },
    }


def test_poolside_config_is_exact_and_has_six_aux_norms():
    config = ModelConfig.from_dict(_config_dict())

    assert config.hidden_size == 3072
    assert config.num_hidden_layers == 6
    assert config.num_attention_heads == 72
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.layer_types == ["sliding_attention"] * 6
    assert config.sliding_window == 512
    assert config.target_layer_ids == [1, 10, 19, 29, 38, 47]
    assert config.num_target_layers == 48
    assert config.mask_token_id == 12
    assert config.vocab_size == config.draft_vocab_size == 100352
    assert config.block_size == 16
    assert len(config.aux_hidden_state_layer_ids) == 6


def test_checkpoint_values_drive_the_contract_and_weight_shapes():
    params = _config_dict()
    params.update(
        {
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 100480,
            "draft_vocab_size": 100480,
            "layer_types": ["sliding_attention"] * 4,
            "sliding_windows": [1024] * 4,
            "sliding_window": 1024,
            "eagle_aux_hidden_state_layer_ids": [2, 18, 34, 50],
        }
    )
    params["dflash_config"].update(
        {
            "block_size": 8,
            "mask_token_id": 17,
            "num_target_layers": 64,
            "target_layer_ids": [1, 17, 33, 49],
        }
    )

    config = ModelConfig.from_dict(params)

    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 4
    assert config.target_layer_ids == [1, 17, 33, 49]
    assert expected_laguna_dflash_weight_shapes(config)["fc.weight"] == (4096, 16384)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda params: params.pop("hidden_size"),
            "missing checkpoint fields: hidden_size",
        ),
        (
            lambda params: params["dflash_config"].pop("target_layer_ids"),
            "missing dflash_config fields: target_layer_ids",
        ),
        (
            lambda params: params.update({"layer_types": ["full_attention"] * 6}),
            "sliding_attention",
        ),
        (
            lambda params: params["dflash_config"].update(
                {"target_layer_ids": [1, 10, 19, 29, 38, 48]}
            ),
            "target_layer_ids",
        ),
    ],
)
def test_malformed_checkpoint_contract_is_rejected(mutate, message):
    params = _config_dict()
    mutate(params)

    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict(params)


def test_target_layer_count_and_tokenizer_length_are_checked():
    config = ModelConfig.from_dict(_config_dict())
    target = SimpleNamespace(num_hidden_layers=48, vocab_size=100352)

    validate_laguna_dflash_target(
        config,
        target_model_config=target,
        target_tokenizer_length=100352,
    )
    with pytest.raises(ValueError, match="layer count"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(num_hidden_layers=47),
            target_tokenizer_length=100352,
        )
    with pytest.raises(ValueError, match="vocabulary"):
        validate_laguna_dflash_target(
            config,
            target_model_config=target,
            target_tokenizer_length=100351,
        )


def test_generic_drafter_gate_invokes_laguna_target_validation():
    from mlx_vlm.speculative.drafters.laguna_dflash import LagunaDFlashDraftModel

    draft = LagunaDFlashDraftModel(ModelConfig.from_dict(_config_dict()))
    target = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=48, vocab_size=100352),
        model=SimpleNamespace(layers=[object()] * 48),
        rollback_speculative_cache=lambda *args: 0,
    )

    validate_drafter_compatibility(target, draft, "dflash")
    target.model.layers.pop()
    with pytest.raises(ValueError, match="layer count"):
        validate_drafter_compatibility(target, draft, "dflash")


def test_generic_model_loader_selects_laguna_dflash_backend():
    from mlx_vlm.utils import get_model_and_args

    architecture, model_type = get_model_and_args(_config_dict())

    assert model_type == "laguna_dflash"
    assert architecture.Model.__name__ == "LagunaDFlashDraftModel"


def test_published_weight_keys_and_shapes_are_explicit():
    config = ModelConfig.from_dict(_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    assert expected["layers.0.self_attn.o_proj.weight"] == (3072, 9216)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}

    validate_laguna_dflash_weights(weights, config)
    bad = dict(weights)
    bad["layers.0.self_attn.g_proj.weight"] = SimpleNamespace(shape=(71, 3072))
    with pytest.raises(ValueError, match="weight shapes"):
        validate_laguna_dflash_weights(bad, config)


def test_quantized_weight_auxiliaries_are_accepted():
    config = ModelConfig.from_dict(_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    weights["fc.weight"] = SimpleNamespace(shape=(3072, 9216))
    weights["fc.scales"] = SimpleNamespace(shape=(3072, 288))
    weights["fc.biases"] = SimpleNamespace(shape=(3072, 288))

    validate_laguna_dflash_weights(weights, config)


def test_weight_key_drift_is_rejected():
    config = ModelConfig.from_dict(_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    weights["layers.0.self_attn.extra.weight"] = SimpleNamespace(shape=(1,))

    with pytest.raises(ValueError, match="weight keys"):
        validate_laguna_dflash_weights(weights, config)


def test_target_without_rollback_support_is_rejected():
    config = ModelConfig.from_dict(_config_dict())

    with pytest.raises(ValueError, match="rollback"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(
                num_hidden_layers=48, vocab_size=100352
            ),
            target_tokenizer_length=100352,
            target_language_model=SimpleNamespace(),
        )
