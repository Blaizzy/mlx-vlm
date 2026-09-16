"""DFlash2, Laguna, and Muse Glimmer drafter contracts."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.muse_glimmer import Model as MuseGlimmerModel
from mlx_vlm.models.muse_glimmer import ModelConfig as MuseGlimmerConfig
from mlx_vlm.models.muse_glimmer import TextConfig, VisionConfig
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import TextConfig as Qwen3_5TextConfig
from mlx_vlm.server.generation import _PositionedTargetSampler
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.dflash2 import ModelConfig as DFlash2Config
from mlx_vlm.speculative.drafters.laguna_dflash import ModelConfig as LagunaDFlashConfig
from mlx_vlm.speculative.drafters.laguna_dflash.config import (
    expected_laguna_dflash_weight_shapes,
    validate_laguna_dflash_target,
    validate_laguna_dflash_weights,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    Model as MuseGlimmerAssistantModel,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    ModelConfig as MuseGlimmerAssistantConfig,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    expected_muse_glimmer_assistant_weight_shapes,
    validate_muse_glimmer_assistant_weights,
)
from mlx_vlm.utils import get_model_and_args

# DFlash2


def _dflash2_published_config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "is_causal": False,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "num_target_layers": 64,
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 2048,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
    }


def _dflash2_config():
    config = _dflash2_published_config()
    config.update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        max_position_embeddings=128,
        num_target_layers=2,
        layer_types=["full_attention"],
        sliding_window=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
    )
    config["dflash_config"] = {
        "block_size": 3,
        "runtime_block_size": 3,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 4,
        "selector_top_k": 4,
        "target_layer_ids": [0],
    }
    return DFlash2Config.from_dict(config)


def _dflash2_target():
    config = Qwen3_5TextConfig(
        model_type="qwen3_5_text",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        head_dim=8,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    outer_config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=config,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=30,
        video_token_id=29,
        vision_start_token_id=28,
    )
    model = qwen_language.LanguageModel(config, outer_config)
    model.set_dtype(mx.bfloat16)
    return model


def _dflash2_tokens(target, prompt, drafter=None, temperature=0, seed=None):
    def get_input_embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
        del pixel_values, kwargs
        position_ids, rope_deltas = target.get_rope_index(
            input_ids, attention_mask=mask
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=target.model.embed_tokens(input_ids),
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    generation_target = SimpleNamespace(
        language_model=target, get_input_embeddings=get_input_embeddings
    )
    kwargs = (
        {"draft_model": drafter, "draft_kind": "dflash"} if drafter is not None else {}
    )
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            prompt,
            generation_target,
            None,
            None,
            max_tokens=10,
            temperature=temperature,
            seed=seed,
            prefill_step_size=None,
            **kwargs,
        )
    ]


def test_published_dflash2_config_and_loader_routing(tmp_path):
    published = _dflash2_published_config()
    config = DFlash2Config.from_dict(published)
    architecture, model_type = get_model_and_args(published)

    assert config.model_type == "dflash2"
    assert config.backbone_model_type == "qwen3"
    assert config.block_size == 8
    assert config.runtime_block_size == 5
    assert config.target_layer_ids == [5, 19, 33, 47, 61]
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.rope_theta == 10000000
    assert config.rope_scaling == {"rope_type": "default"}
    assert model_type == "dflash2"
    assert architecture.Model is DFlash2DraftModel

    drafter = DFlash2DraftModel(config)
    assert drafter.prefer_requested_block_size is False
    assert drafter.dflash_initial_block_size == 3
    assert drafter.dflash_min_block_size == 3

    (tmp_path / "config.json").write_text(json.dumps(published))
    assert resolve_drafter_kind(tmp_path) == "dflash"


def test_dflash2_sanitize_normalizes_published_codebooks():
    drafter = DFlash2DraftModel(_dflash2_config())
    predecessor = mx.zeros((32, 4))
    successor = mx.ones((32, 4))

    weights = drafter.sanitize(
        {
            "candidate_selector.predecessor_codebook": predecessor,
            "candidate_selector.successor_codebook": successor,
        }
    )

    assert weights == {
        "candidate_selector.predecessor_codebook.weight": predecessor,
        "candidate_selector.successor_codebook.weight": successor,
    }


def test_positioned_proposal_sampling_is_independent_of_target_filters():
    sampler = _PositionedTargetSampler(temperature=1.0, top_p=0.95, top_k=20, seed=7)
    scores = mx.zeros((1, 16))

    first = sampler.sample_proposal(scores, row_ids=[0], positions=[3])
    second = sampler.sample_proposal(scores, row_ids=[0], positions=[3])

    assert first.shape == (1,)
    assert bool(mx.array_equal(first, second))


@pytest.mark.parametrize(("temperature", "seed"), [(0, None), (1.0, 17)])
def test_dflash2_generation_has_exact_target_parity(temperature, seed):
    mx.random.seed(7)
    target = _dflash2_target()
    drafter = DFlash2DraftModel(_dflash2_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _dflash2_tokens(target, prompt, temperature=temperature, seed=seed)
    speculative = _dflash2_tokens(
        target, prompt, drafter, temperature=temperature, seed=seed
    )

    assert speculative == baseline
    assert drafter.draft_lens


# Laguna


def _laguna_config_dict():
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
    params = _laguna_config_dict()
    mutate(params)

    with pytest.raises(ValueError, match=message):
        LagunaDFlashConfig.from_dict(params)


def test_target_layer_count_and_tokenizer_length_are_checked():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    target = SimpleNamespace(num_hidden_layers=48, vocab_size=100352)

    validate_laguna_dflash_target(
        config, target_model_config=target, target_tokenizer_length=100352
    )
    with pytest.raises(ValueError, match="layer count"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(num_hidden_layers=47),
            target_tokenizer_length=100352,
        )
    with pytest.raises(ValueError, match="vocabulary"):
        validate_laguna_dflash_target(
            config, target_model_config=target, target_tokenizer_length=100351
        )


def test_generic_drafter_gate_invokes_laguna_target_validation():
    from mlx_vlm.speculative.drafters.laguna_dflash import LagunaDFlashDraftModel

    draft = LagunaDFlashDraftModel(LagunaDFlashConfig.from_dict(_laguna_config_dict()))
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

    architecture, model_type = get_model_and_args(_laguna_config_dict())

    assert model_type == "laguna_dflash"
    assert architecture.Model.__name__ == "LagunaDFlashDraftModel"


def test_published_weight_keys_and_shapes_are_explicit():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    assert expected["layers.0.self_attn.o_proj.weight"] == (3072, 9216)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}

    validate_laguna_dflash_weights(weights, config)
    bad = dict(weights)
    bad["layers.0.self_attn.g_proj.weight"] = SimpleNamespace(shape=(71, 3072))
    with pytest.raises(ValueError, match="weight shapes"):
        validate_laguna_dflash_weights(bad, config)


def test_weight_key_drift_is_rejected():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    weights["layers.0.self_attn.extra.weight"] = SimpleNamespace(shape=(1,))

    with pytest.raises(ValueError, match="weight keys"):
        validate_laguna_dflash_weights(weights, config)


def test_target_without_rollback_support_is_rejected():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())

    with pytest.raises(ValueError, match="rollback"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(
                num_hidden_layers=48, vocab_size=100352
            ),
            target_tokenizer_length=100352,
            target_language_model=SimpleNamespace(),
        )


# Muse Glimmer


def _glimmer_published_config():
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
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 2048,
        "block_size": 16,
        "mask_token_id": 201818,
        "target_layer_ids": [1, 13, 25, 37, 49],
    }


def _tiny_assistant_config():
    return MuseGlimmerAssistantConfig(
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


def _glimmer_target():
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


def _glimmer_tokens(target, prompt, drafter=None):
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
    config = MuseGlimmerAssistantConfig.from_dict(_glimmer_published_config())

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
        (lambda config: config.update({"mask_token_id": 202048}), "mask_token_id"),
    ],
)
def test_invalid_checkpoint_contract_is_rejected(mutate, message):
    config = _glimmer_published_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        MuseGlimmerAssistantConfig.from_dict(config)


def test_binding_uses_raw_target_embedding_and_checks_target_family():
    target = _glimmer_target()
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


def test_greedy_speculative_generation_matches_baseline():
    mx.random.seed(7)
    target = _glimmer_target()
    drafter = MuseGlimmerAssistantModel(_tiny_assistant_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    baseline = _glimmer_tokens(target, prompt)
    speculative = _glimmer_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens
