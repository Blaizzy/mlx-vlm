import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import TextConfig as Qwen3_5TextConfig
from mlx_vlm.server.generation import _PositionedTargetSampler
from mlx_vlm.speculative.drafters import (
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    ModelConfig,
    _grouped_dynamic_convolve,
)
from mlx_vlm.utils import get_model_and_args


def _published_config():
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
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
        },
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


def _tiny_config():
    config = _published_config()
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
    return ModelConfig.from_dict(config)


def _tiny_target():
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


def _generated_tokens(target, prompt, drafter=None, temperature=0, seed=None):
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
    published = _published_config()
    config = ModelConfig.from_dict(published)
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
    drafter = DFlash2DraftModel(_tiny_config())
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


def test_grouped_dynamic_convolution_is_causal():
    hidden = mx.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
    dynamic = mx.zeros((1, 3, 2, 2))
    base = mx.array([[1, 1, 1, 1], [2, 2, 2, 2]])

    actual = _grouped_dynamic_convolve(hidden, dynamic, base, group_size=2)
    expected = mx.array([[[1, 2, 3, 4], [7, 10, 13, 16], [19, 22, 25, 28]]])

    assert bool(mx.array_equal(actual, expected))


def test_candidate_selector_uses_coherent_top_candidate_path():
    config = _tiny_config()
    selector = CandidateSelector(config)
    selector.predecessor_codebook.weight = mx.zeros((32, 4))
    selector.successor_codebook.weight = mx.zeros((32, 4))
    selector.hidden_projection.weight = mx.zeros((4, 16))
    hidden = mx.zeros((1, 2, 16))
    logits = mx.array([[[0, 1, 5, 2] + [0] * 28, [0, 7, 2, 3] + [0] * 28]])

    selected = selector.select(
        hidden,
        logits,
        mx.array([4]),
        sampler=lambda scores: mx.argmax(scores, axis=-1),
    )

    assert selected.tolist() == [[2, 1]]


def test_positioned_proposal_sampling_is_independent_of_target_filters():
    sampler = _PositionedTargetSampler(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=7,
    )
    scores = mx.zeros((1, 16))

    first = sampler.sample_proposal(scores, row_ids=[0], positions=[3])
    second = sampler.sample_proposal(scores, row_ids=[0], positions=[3])

    assert first.shape == (1,)
    assert bool(mx.array_equal(first, second))


def test_dflash2_target_validation_is_structural():
    config = _tiny_config()
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=16,
                num_hidden_layers=2,
                vocab_size=32,
            ),
            model=SimpleNamespace(layers=[object(), object()]),
            rollback_speculative_cache=lambda *args: None,
        )
    )

    validate_drafter_compatibility(
        target, DFlash2DraftModel(config), draft_kind="dflash"
    )


@pytest.mark.parametrize(("temperature", "seed"), [(0, None), (1.0, 17)])
def test_dflash2_generation_has_exact_target_parity(temperature, seed):
    mx.random.seed(7)
    target = _tiny_target()
    drafter = DFlash2DraftModel(_tiny_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt, temperature=temperature, seed=seed)
    speculative = _generated_tokens(
        target,
        prompt,
        drafter,
        temperature=temperature,
        seed=seed,
    )

    assert speculative == baseline
    assert drafter.draft_lens


def _tiny_glm5_next_target():
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        qk_head_dim=8,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = LanguageModel(config)
    model.set_dtype(mx.bfloat16)
    return model


def _tiny_glm5_next_drafter_config():
    config = _published_config()
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
    return ModelConfig.from_dict(config)


def _glm5_next_generated_tokens(target, prompt, drafter=None, temperature=0, seed=None):
    def get_input_embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
        del pixel_values, mask, kwargs
        return InputEmbeddingsFeatures(
            inputs_embeds=target.model.embed_tokens(input_ids)
        )

    generation_target = SimpleNamespace(
        language_model=target,
        get_input_embeddings=get_input_embeddings,
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


def test_dflash2_glm5_next_generation_has_exact_target_parity():
    mx.random.seed(7)
    target = _tiny_glm5_next_target()
    drafter = DFlash2DraftModel(_tiny_glm5_next_drafter_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[2, 4, 6, 8]], dtype=mx.int32)

    baseline = _glm5_next_generated_tokens(target, prompt)
    speculative = _glm5_next_generated_tokens(target, prompt, drafter)

    assert speculative == baseline
    assert drafter.draft_lens


def test_dflash2_glm5_next_batch_generation_matches_target():
    from mlx_vlm.generate import BatchGenerator

    class NeverStop:
        def __call__(self, _token):
            return False

        def add_eos_token_ids(self, _tokens):
            return None

    def generate(target, prompts, max_tokens, drafter=None):
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(stopping_criteria=NeverStop())
        )
        generator = BatchGenerator(
            target,
            processor,
            prefill_batch_size=2,
            completion_batch_size=2,
            prefill_step_size=None,
            draft_model=drafter,
            draft_kind="dflash" if drafter is not None else None,
            draft_block_size=3,
            greedy_sampling=True,
        )
        prompt_kwargs = [
            {
                "inputs_embeds": target.model.embed_tokens(
                    mx.array([prompt], dtype=mx.int32)
                )
            }
            for prompt in prompts
        ]
        uids = generator.insert(
            prompts, max_tokens=max_tokens, prompt_kwargs=prompt_kwargs
        )
        tokens = {uid: [] for uid in uids}
        try:
            for _ in range(32):
                if not generator.has_work:
                    break
                _, responses = generator.next()
                for response in responses:
                    if response.token is not None:
                        tokens[response.uid].append(int(response.token))
            assert not generator.has_work
        finally:
            generator.close()
        return [tokens[uid] for uid in uids]

    mx.random.seed(0)
    target = _tiny_glm5_next_target()
    mx.eval(target.parameters())
    prompts = [[2, 4, 6], [3, 5, 7, 9, 11]]
    max_tokens = [3, 5]

    expected = generate(target, prompts, max_tokens)
    drafter = DFlash2DraftModel(_tiny_glm5_next_drafter_config())
    mx.eval(drafter.parameters())
    actual = generate(target, prompts, max_tokens, drafter=drafter)

    assert actual == expected
    assert [len(row) for row in actual] == max_tokens
