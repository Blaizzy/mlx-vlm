import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.exact_speculative_verify import exact_speculative_verify_weight
from mlx_vlm.models.lfm2 import Model as Lfm2Model
from mlx_vlm.models.lfm2 import ModelConfig as Lfm2Config
from mlx_vlm.models.lfm2.language import Lfm2MoeSparseMoeBlock
from mlx_vlm.models.lfm2.speculative_verifier import Lfm2ExactSpeculativeVerifier
from mlx_vlm.models.lfm2_moe import Model as Lfm2MoeModel
from mlx_vlm.models.lfm2_moe import ModelConfig as Lfm2MoeConfig
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import TextConfig as Qwen3_5TextConfig
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.dspark import (
    DSparkDraftModel,
    ModelConfig,
    validate_dspark_target,
)
from mlx_vlm.utils import get_model_and_args


def _published_config():
    return {
        "architectures": ["Lfm2DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 6144,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "vocab_size": 128000,
        "rope_theta": 10000000.0,
        "max_position_embeddings": 128000,
        "layer_types": ["full_attention"] * 5,
        "block_size": 9,
        "dflash_config": {
            "mask_token_id": 125017,
            "target_layer_ids": [2, 9, 17, 21, 27],
            "num_target_layers": 30,
        },
        "markov_rank": 256,
        "rope_is_neox_style": False,
        "enable_confidence_head": True,
        "markov_head_type": "vanilla",
    }


def _published_qwen38_config():
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "block_size": 7,
        "confidence_head_with_markov": True,
        "hidden_size": 5120,
        "intermediate_size": 10240,
        "num_hidden_layers": 5,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "num_target_layers": 64,
        "layer_types": ["full_attention"] * 5,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "dflash_config": {
            "projector_type": "dspark",
            "mask_token_id": 248077,
            "target_layer_ids": [4, 16, 28, 40, 52],
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
        },
    }


def _tiny_draft_config():
    return ModelConfig.from_dict(
        {
            "architectures": ["Lfm2DSparkDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "rope_is_neox_style": False,
            "max_position_embeddings": 128,
            "layer_types": ["full_attention"],
            "block_size": 3,
            "dflash_config": {
                "mask_token_id": 31,
                "target_layer_ids": [0, 2],
                "num_target_layers": 3,
            },
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
        }
    )


def _tiny_qwen38_draft_config():
    return ModelConfig.from_dict(
        {
            "architectures": ["DSparkDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "layer_types": ["full_attention"],
            "block_size": 3,
            "num_target_layers": 2,
            "dflash_config": {
                "projector_type": "dspark",
                "mask_token_id": 31,
                "target_layer_ids": [0],
            },
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
        }
    )


def _published_moe_config():
    config = _published_config()
    config["rope_theta"] = 5000000.0
    config["dflash_config"] = {
        "mask_token_id": 125017,
        "target_layer_ids": [2, 6, 10, 14, 18],
        "num_target_layers": 24,
    }
    return config


def _published_1_2b_config():
    config = _published_config()
    config["vocab_size"] = 65536
    config["rope_theta"] = 1000000.0
    config["dflash_config"] = {
        "mask_token_id": 64402,
        "target_layer_ids": [2, 5, 8, 11, 14],
        "num_target_layers": 16,
    }
    return config


def _tiny_target():
    config = Lfm2Config(
        model_type="lfm2",
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        block_dim=8,
        block_ff_dim=16,
        block_multiple_of=1,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        rope_theta=10000.0,
        layer_types=["conv", "full_attention", "conv"],
        full_attn_idxs=[1],
        tie_word_embeddings=True,
    )
    return Lfm2Model(config)


def _tiny_moe_target():
    config = Lfm2MoeConfig(
        model_type="lfm2_moe",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        use_expert_bias=True,
        num_dense_layers=1,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        rope_theta=10000.0,
        layer_types=["conv", "full_attention", "conv"],
        tie_word_embeddings=True,
    )
    return Lfm2MoeModel(config)


def _tiny_qwen38_target():
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


def _generated_tokens(
    target,
    prompt,
    drafter=None,
    *,
    max_tokens=10,
    temperature=0,
    seed=None,
):
    kwargs = {}
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="dflash")
    if hasattr(target, "language_model"):
        generation_target = target
    else:

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
            language_model=target,
            get_input_embeddings=get_input_embeddings,
        )
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            prompt,
            generation_target,
            None,
            None,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            prefill_step_size=None,
            **kwargs,
        )
    ]


def test_published_config_normalizes_dspark_gamma_to_verify_width():
    config = ModelConfig.from_dict(_published_config())
    drafter = DSparkDraftModel(config)

    assert config.model_type == "dspark"
    assert config.backbone_model_type == "qwen3"
    assert config.proposal_length == 9
    assert config.block_size == 10
    assert config.runtime_block_size == 8
    assert config.target_layer_ids == [2, 9, 17, 21, 27]
    assert config.num_target_layers == 30
    assert config.markov_rank == 256
    assert config.block_size_policy == "fixed"
    assert drafter.prefer_requested_block_size is True


def test_published_qwen38_config_normalizes_dspark_contract_and_yarn():
    config = ModelConfig.from_dict(_published_qwen38_config())
    drafter = DSparkDraftModel(config)

    assert config.proposal_length == 7
    assert config.block_size == 8
    assert config.runtime_block_size == 8
    assert config.target_layer_ids == [4, 16, 28, 40, 52]
    assert config.num_target_layers == 64
    assert config.mask_token_id == 248077
    assert config.rope_theta == 10000000
    assert config.rope_scaling == {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    assert config.rope_is_neox_style is True
    assert config.block_size_policy == "adaptive"
    assert config.dflash_initial_block_size == 4
    assert drafter.prefer_requested_block_size is False
    assert drafter.rope.traditional is False


def test_dspark_config_uses_explicit_rope_capability_not_architecture_name():
    published = _published_config()
    published.pop("rope_is_neox_style")
    published["architectures"] = ["LfmNamedFutureDSparkModel"]

    config = ModelConfig.from_dict(published)

    assert config.rope_is_neox_style is True


def test_dspark_block_policy_can_be_declared_by_any_checkpoint():
    published = _published_config()
    published["dflash_config"]["block_size_policy"] = "adaptive"
    published["dflash_config"]["dflash_initial_block_size"] = 3

    config = ModelConfig.from_dict(published)
    drafter = DSparkDraftModel(config)

    assert config.block_size_policy == "adaptive"
    assert config.dflash_initial_block_size == 3
    assert drafter.prefer_requested_block_size is False


def test_published_lfm2_moe_dspark_config_preserves_target_layers():
    config = ModelConfig.from_dict(_published_moe_config())

    assert config.proposal_length == 9
    assert config.block_size == 10
    assert config.target_layer_ids == [2, 6, 10, 14, 18]
    assert config.num_target_layers == 24
    assert config.rope_theta == 5000000.0


def test_published_1_2b_dspark_config_preserves_target_contract():
    config = ModelConfig.from_dict(_published_1_2b_config())

    assert config.proposal_length == 9
    assert config.block_size == 10
    assert config.runtime_block_size == 8
    assert config.target_layer_ids == [2, 5, 8, 11, 14]
    assert config.num_target_layers == 16
    assert config.vocab_size == 65536
    assert config.mask_token_id == 64402
    assert config.rope_theta == 1000000.0


def test_published_1_2b_dspark_accepts_matching_target_metadata():
    config = ModelConfig.from_dict(_published_1_2b_config())
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(
                model_type="lfm2",
                hidden_size=2048,
                num_hidden_layers=16,
                vocab_size=65536,
            ),
            model=SimpleNamespace(layers=[object()] * 16),
            rollback_speculative_cache=lambda *args: None,
        )
    )

    validate_dspark_target(config, target)


def test_generic_loader_routes_markov_dflash_checkpoint_to_dspark(tmp_path):
    architecture, model_type = get_model_and_args(_published_config())

    assert model_type == "dspark"
    assert architecture.Model is DSparkDraftModel

    (tmp_path / "config.json").write_text(json.dumps(_published_config()))
    assert resolve_drafter_kind(tmp_path) == "dflash"


def test_generic_loader_routes_qwen38_dspark_checkpoint(tmp_path):
    published = _published_qwen38_config()
    architecture, model_type = get_model_and_args(published)

    assert model_type == "dspark"
    assert architecture.Model is DSparkDraftModel

    (tmp_path / "config.json").write_text(json.dumps(published))
    assert resolve_drafter_kind(tmp_path) == "dflash"


def test_generic_loader_routes_nested_markov_contract_to_dspark():
    published = _published_qwen38_config()
    published.pop("markov_rank")

    architecture, model_type = get_model_and_args(published)

    assert model_type == "dspark"
    assert architecture.Model is DSparkDraftModel


def test_dspark_requires_matching_target_structure():
    drafter = DSparkDraftModel(_tiny_draft_config())
    target = _tiny_target()

    validate_drafter_compatibility(target, drafter, "dflash")
    target.language_model.config.hidden_size += 1
    with pytest.raises(ValueError, match="target hidden-size mismatch"):
        validate_drafter_compatibility(target, drafter, "dflash")


def test_dspark_target_validation_is_family_agnostic():
    config = _tiny_draft_config()
    target = _tiny_target()
    target.language_model.config.model_type = "future_hybrid_model"

    validate_dspark_target(config, target)


def test_dspark_rollback_error_names_target_model_class():
    class TargetWithoutRollback:
        def __init__(self):
            self.config = SimpleNamespace(
                hidden_size=8,
                num_hidden_layers=3,
                vocab_size=32,
            )
            self.model = SimpleNamespace(layers=[object()] * 3)

    target = SimpleNamespace(language_model=TargetWithoutRollback())

    with pytest.raises(
        ValueError,
        match="DSpark target TargetWithoutRollback does not expose",
    ):
        validate_dspark_target(_tiny_draft_config(), target)


def test_dspark_accepts_matching_lfm2_moe_target():
    drafter = DSparkDraftModel(_tiny_draft_config())

    validate_drafter_compatibility(_tiny_moe_target(), drafter, "dflash")


def test_published_qwen38_dspark_accepts_nested_target_metadata():
    config = ModelConfig.from_dict(_published_qwen38_config())
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3_5",
                text_config=SimpleNamespace(
                    model_type="qwen3_5_text",
                    hidden_size=5120,
                    num_hidden_layers=64,
                    vocab_size=248320,
                ),
            ),
            model=SimpleNamespace(layers=[object()] * 64),
            rollback_speculative_cache=lambda *args: None,
        )
    )

    validate_dspark_target(config, target)


def test_dspark_accepts_matching_qwen38_target():
    drafter = DSparkDraftModel(_tiny_qwen38_draft_config())

    validate_drafter_compatibility(_tiny_qwen38_target(), drafter, "dflash")


def test_tiny_dspark_forward_uses_markov_head_and_published_block_semantics():
    mx.random.seed(0)
    target = _tiny_target()
    drafter = DSparkDraftModel(_tiny_draft_config())
    mx.eval(target.parameters(), drafter.parameters())

    assert drafter.rope.traditional is True
    target_cache = target.make_cache()
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    output = target.language_model(
        prompt,
        cache=target_cache,
        capture_layer_ids=drafter.config.target_layer_ids,
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    draft_cache = drafter.reset(target)
    tokens = drafter.draft_block(
        4,
        hidden,
        draft_cache,
        block_size=drafter.config.block_size,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    mx.eval(tokens)

    assert hidden.shape == (1, 3, 16)
    assert tokens.shape == (1, drafter.config.proposal_length)
    assert all(cache.offset == 3 for cache in draft_cache)


def test_exact_speculative_verify_dense_kernel_matches_mlx_linear():
    if not mx.metal.is_available():
        pytest.skip("The target-verification kernel requires Metal.")

    mx.random.seed(11)
    inputs = mx.random.normal((1, 5, 8)).astype(mx.bfloat16)
    weight = mx.random.normal((16, 8)).astype(mx.bfloat16)
    expected = inputs @ weight.T
    actual = exact_speculative_verify_weight(weight, inputs)
    assert actual is not None
    mx.eval(expected, actual)

    assert actual.shape == expected.shape
    assert bool(mx.array_equal(actual, expected))


def test_lfm2_moe_exact_speculative_verify_narrow_router_matches_singletons():
    mx.random.seed(12)
    moe = Lfm2MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            use_expert_bias=True,
        )
    )
    moe.set_dtype(mx.bfloat16)
    inputs = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [moe(inputs[:, position : position + 1]) for position in range(5)],
        axis=1,
    )
    actual = Lfm2ExactSpeculativeVerifier()._feed_forward(moe, inputs)
    mx.eval(expected, actual)

    assert bool(mx.array_equal(actual, expected))


def test_lfm2_exact_speculative_verify_matches_single_token_decode():
    if not mx.metal.is_available():
        pytest.skip("Exact target verification requires Metal.")

    mx.random.seed(13)
    target = _tiny_target()
    lm = target.language_model
    mx.eval(target.parameters())

    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    verify = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    block_cache = lm.make_cache()
    singleton_cache = lm.make_cache()
    lm(prompt, cache=block_cache)
    lm(prompt, cache=singleton_cache)

    block_logits = lm(
        verify,
        cache=block_cache,
        speculative_verify=True,
    ).logits
    singleton_logits = mx.concatenate(
        [
            lm(verify[:, index : index + 1], cache=singleton_cache).logits
            for index in range(verify.shape[1])
        ],
        axis=1,
    )
    mx.eval(block_logits, singleton_logits)

    assert bool(mx.array_equal(block_logits, singleton_logits))


def test_lfm2_moe_exact_speculative_verify_matches_single_token_decode():
    if not mx.metal.is_available():
        pytest.skip("Exact target verification requires Metal.")

    mx.random.seed(17)
    target = _tiny_moe_target()
    target.set_dtype(mx.bfloat16)
    lm = target.language_model
    mx.eval(target.parameters())

    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    verify = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    block_cache = lm.make_cache()
    singleton_cache = lm.make_cache()
    lm(prompt, cache=block_cache)
    lm(prompt, cache=singleton_cache)

    block_logits = lm(
        verify,
        cache=block_cache,
        speculative_verify=True,
    ).logits
    singleton_logits = mx.concatenate(
        [
            lm(verify[:, index : index + 1], cache=singleton_cache).logits
            for index in range(verify.shape[1])
        ],
        axis=1,
    )
    mx.eval(block_logits, singleton_logits)

    assert bool(mx.array_equal(block_logits, singleton_logits))


def test_qwen38_exact_speculative_verify_matches_single_token_decode():
    if not mx.metal.is_available():
        pytest.skip("Exact target verification requires Metal.")

    mx.random.seed(23)
    lm = _tiny_qwen38_target()
    mx.eval(lm.parameters())

    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    verify = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    block_cache = lm.make_cache()
    singleton_cache = lm.make_cache()
    lm(prompt, cache=block_cache)
    lm(prompt, cache=singleton_cache)

    block_logits = lm(
        verify,
        cache=block_cache,
        capture_layer_ids=[0],
        speculative_verify=True,
    ).logits
    singleton_logits = mx.concatenate(
        [
            lm(verify[:, index : index + 1], cache=singleton_cache).logits
            for index in range(verify.shape[1])
        ],
        axis=1,
    )
    mx.eval(block_logits, singleton_logits)

    assert bool(mx.array_equal(block_logits, singleton_logits))


def test_qwen38_draft_prefill_capture_keeps_ordinary_target_path():
    mx.random.seed(25)
    lm = _tiny_qwen38_target()
    mx.eval(lm.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = lm(prompt, cache=lm.make_cache())
    captured = lm(prompt, cache=lm.make_cache(), capture_layer_ids=[0])
    mx.eval(baseline.logits, captured.logits, *captured.hidden_states)

    assert bool(mx.array_equal(baseline.logits, captured.logits))
    assert captured.gdn_states is None
    assert len(captured.hidden_states) == 1


@pytest.mark.parametrize("target_factory", [_tiny_target, _tiny_moe_target])
def test_lfm2_hybrid_cache_rollback_matches_committed_prefix(target_factory):
    mx.random.seed(3)
    target = target_factory()
    lm = target.language_model
    mx.eval(target.parameters())

    prompt = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    verify = mx.array([[6, 7, 8, 9]], dtype=mx.int32)
    accepted = 1

    rolled_cache = lm.make_cache()
    lm(prompt, cache=rolled_cache)
    verify_out = lm(
        verify,
        cache=rolled_cache,
        capture_layer_ids=[0, 2],
        speculative_verify=True,
    )
    lm.rollback_speculative_cache(
        rolled_cache,
        verify_out.gdn_states,
        accepted,
        block_size=verify.shape[1],
    )

    reference_cache = lm.make_cache()
    committed = mx.concatenate([prompt, verify[:, : accepted + 1]], axis=1)
    lm(committed, cache=reference_cache)

    probe = mx.array([[10]], dtype=mx.int32)
    rolled_logits = lm(probe, cache=rolled_cache).logits
    reference_logits = lm(probe, cache=reference_cache).logits
    mx.eval(rolled_logits, reference_logits)

    assert rolled_cache[1].offset == reference_cache[1].offset
    assert bool(mx.allclose(rolled_cache[0][0], reference_cache[0][0], atol=1e-5))
    assert bool(mx.allclose(rolled_cache[2][0], reference_cache[2][0], atol=1e-5))
    assert bool(mx.allclose(rolled_logits, reference_logits, atol=1e-4))


def test_lfm2_ragged_batch_rollback_matches_each_committed_prefix():
    mx.random.seed(5)
    target = _tiny_target()
    lm = target.language_model
    mx.eval(target.parameters())

    prompt = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
    verify = mx.array([[9, 10, 11, 12], [13, 14, 15, 16]], dtype=mx.int32)
    accepted = mx.array([0, 2], dtype=mx.int32)
    rolled_cache = [
        BatchKVCache([0, 0]) if layer.is_attention_layer else ArraysCache(size=1)
        for layer in lm.model.layers
    ]

    lm(prompt, cache=rolled_cache)
    verify_out = lm(verify, cache=rolled_cache, speculative_verify=True)
    lm.rollback_speculative_cache(
        rolled_cache,
        verify_out.gdn_states,
        accepted,
        block_size=verify.shape[1],
    )
    assert rolled_cache[1].offset.tolist() == [5, 7]
    probes = mx.array([[17], [18]], dtype=mx.int32)
    rolled_logits = lm(probes, cache=rolled_cache).logits

    reference_logits = []
    for row, accepted_count in enumerate(accepted.tolist()):
        reference_cache = lm.make_cache()
        committed = mx.concatenate(
            [prompt[row], verify[row, : int(accepted_count) + 1]], axis=0
        )[None, :]
        lm(committed, cache=reference_cache)
        reference_logits.append(lm(probes[row : row + 1], cache=reference_cache).logits)
    reference_logits = mx.concatenate(reference_logits, axis=0)
    mx.eval(rolled_logits, reference_logits)

    assert bool(mx.allclose(rolled_logits, reference_logits, atol=1e-4))


def test_greedy_dspark_generation_matches_lfm2_baseline():
    mx.random.seed(7)
    target = _tiny_target()
    drafter = DSparkDraftModel(_tiny_draft_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    speculative = _generated_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens


def test_greedy_dspark_generation_matches_lfm2_moe_baseline():
    mx.random.seed(19)
    target = _tiny_moe_target()
    drafter = DSparkDraftModel(_tiny_draft_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    speculative = _generated_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens


def test_greedy_dspark_generation_matches_qwen38_baseline():
    mx.random.seed(29)
    target = _tiny_qwen38_target()
    drafter = DSparkDraftModel(_tiny_qwen38_draft_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    speculative = _generated_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens


@pytest.mark.parametrize(
    ("target_factory", "draft_config_factory"),
    [
        (_tiny_target, _tiny_draft_config),
        (_tiny_qwen38_target, _tiny_qwen38_draft_config),
    ],
)
def test_dspark_repeated_generation_resets_request_state(
    target_factory,
    draft_config_factory,
):
    mx.random.seed(31)
    target = target_factory()
    drafter = DSparkDraftModel(draft_config_factory())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    first = _generated_tokens(target, prompt, drafter)
    first_accept_lens = list(drafter.accept_lens)
    first_draft_lens = list(drafter.draft_lens)

    # These lists drive the adaptive controller and must describe one request,
    # not lifetime state carried over from a previous generation.
    drafter.accept_lens.append(999)
    drafter.draft_lens.append(999)
    second = _generated_tokens(target, prompt, drafter)

    assert first == second == baseline
    assert drafter.accept_lens == first_accept_lens
    assert drafter.draft_lens == first_draft_lens


@pytest.mark.parametrize(
    ("target_factory", "draft_config_factory"),
    [
        (_tiny_target, _tiny_draft_config),
        (_tiny_qwen38_target, _tiny_qwen38_draft_config),
    ],
)
@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0])
def test_sampled_dspark_generation_matches_baseline(
    target_factory,
    draft_config_factory,
    temperature,
):
    mx.random.seed(37)
    target = target_factory()
    drafter = DSparkDraftModel(draft_config_factory())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(
        target,
        prompt,
        max_tokens=24,
        temperature=temperature,
        seed=41,
    )
    speculative = _generated_tokens(
        target,
        prompt,
        drafter,
        max_tokens=24,
        temperature=temperature,
        seed=41,
    )

    assert speculative == baseline
    assert drafter.draft_lens


def test_sampled_dspark_generation_matches_stateful_baseline():
    mx.random.seed(43)
    target = _tiny_qwen38_target()
    drafter = DSparkDraftModel(_tiny_qwen38_draft_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    mx.random.seed(47)
    baseline = _generated_tokens(
        target,
        prompt,
        max_tokens=24,
        temperature=0.7,
    )
    mx.random.seed(47)
    speculative = _generated_tokens(
        target,
        prompt,
        drafter,
        max_tokens=24,
        temperature=0.7,
    )

    assert speculative == baseline
    assert drafter.draft_lens


def test_target_compatibility_rejects_wrong_size():
    drafter = DSparkDraftModel(_tiny_draft_config())
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(
                model_type="lfm2",
                hidden_size=16,
                num_hidden_layers=3,
                vocab_size=32,
            ),
            model=SimpleNamespace(layers=[object()] * 3),
            rollback_speculative_cache=lambda *args: None,
        )
    )

    with pytest.raises(ValueError, match="hidden-size mismatch"):
        validate_drafter_compatibility(target, drafter, "dflash")
