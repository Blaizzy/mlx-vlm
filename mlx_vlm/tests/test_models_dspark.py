import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.exact_speculative_verify import exact_speculative_verify_weight
from mlx_vlm.models.lfm2 import Model as Lfm2Model
from mlx_vlm.models.lfm2 import ModelConfig as Lfm2Config
from mlx_vlm.models.lfm2_moe import Model as Lfm2MoeModel
from mlx_vlm.models.lfm2_moe import ModelConfig as Lfm2MoeConfig
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.qwen3_dspark import DSparkDraftModel, ModelConfig
from mlx_vlm.speculative.utils import run_speculative_rounds
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


def _published_moe_config():
    config = _published_config()
    config["rope_theta"] = 5000000.0
    config["dflash_config"] = {
        "mask_token_id": 125017,
        "target_layer_ids": [2, 6, 10, 14, 18],
        "num_target_layers": 24,
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


def _generated_tokens(target, prompt, drafter=None):
    kwargs = {}
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="dflash")
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
            **kwargs,
        )
    ]


def test_published_config_normalizes_dspark_gamma_to_verify_width():
    config = ModelConfig.from_dict(_published_config())

    assert config.proposal_length == 9
    assert config.block_size == 10
    assert config.runtime_block_size == 8
    assert config.target_layer_ids == [2, 9, 17, 21, 27]
    assert config.num_target_layers == 30
    assert config.markov_rank == 256
    assert DSparkDraftModel.prefer_requested_block_size is True


def test_published_lfm2_moe_dspark_config_preserves_target_layers():
    config = ModelConfig.from_dict(_published_moe_config())

    assert config.proposal_length == 9
    assert config.block_size == 10
    assert config.target_layer_ids == [2, 6, 10, 14, 18]
    assert config.num_target_layers == 24
    assert config.rope_theta == 5000000.0


def test_generic_loader_routes_markov_dflash_checkpoint_to_dspark(tmp_path):
    architecture, model_type = get_model_and_args(_published_config())

    assert model_type == "qwen3_dspark"
    assert architecture.Model is DSparkDraftModel

    (tmp_path / "config.json").write_text(json.dumps(_published_config()))
    assert resolve_drafter_kind(tmp_path) == "dflash"


def test_dspark_requires_the_matching_lfm2_target():
    drafter = DSparkDraftModel(_tiny_draft_config())
    target = _tiny_target()

    validate_drafter_compatibility(target, drafter, "dflash")
    target.language_model.config.model_type = "other"
    with pytest.raises(ValueError, match="requires an LFM2 target"):
        validate_drafter_compatibility(target, drafter, "dflash")


def test_dspark_accepts_matching_lfm2_moe_target():
    drafter = DSparkDraftModel(_tiny_draft_config())

    validate_drafter_compatibility(_tiny_moe_target(), drafter, "dflash")


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


def test_dspark_rejects_non_greedy_speculative_sampling():
    drafter = DSparkDraftModel(_tiny_draft_config())
    rounds = run_speculative_rounds(
        None,
        drafter,
        [],
        mx.array([[1]], dtype=mx.int32),
        mx.array([2], dtype=mx.int32),
        mx.array([0.0]),
        None,
        draft_kind="dflash",
        max_tokens=1,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        sampler_is_greedy=False,
    )

    with pytest.raises(ValueError, match="temperature=0"):
        next(rounds)


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
