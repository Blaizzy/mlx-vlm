"""GLM export compatibility and state preservation through APC batching."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.glm5_next import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel


def tiny_glm():
    return LanguageModel(
        TextConfig(
            model_type="glm5_next_text",
            vocab_size=128,
            hidden_size=128,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=8,
            routed_scaling_factor=2.5,
            kv_lora_rank=64,
            q_lora_rank=128,
            qk_rope_head_dim=0,
            v_head_dim=64,
            qk_nope_head_dim=64,
            num_experts_per_tok=4,
            first_k_dense_replace=1,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            index_topk=6,
            index_head_dim=64,
            index_n_heads=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            mlp_layer_types=["dense", "sparse"],
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 64,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
            },
            index_kpool=3,
            hc_mult=4,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


@pytest.mark.parametrize("bits", [None, 4])
def test_fused_glm_export_preserves_weights_and_logits(bits):
    mx.random.seed(23)
    source = tiny_glm()
    target = tiny_glm()
    if bits is not None:
        nn.quantize(source, bits=bits, group_size=64)
        nn.quantize(target, bits=bits, group_size=64)
    original = dict(tree_flatten(source.parameters()))
    exported = {
        key.replace(".forget_gate.", ".").replace(
            ".self_attn.conv1d.", ".self_attn.qkv_conv.conv."
        ): value
        for key, value in original.items()
    }
    fused_modules = {
        "model.layers.0.self_attn.qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "model.layers.0.self_attn.fbg_a_proj": ("f_a_proj", "b_proj", "g_a_proj"),
        "model.layers.1.self_attn.qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
        "model.layers.0.mlp.gate_up_proj": ("gate_proj", "up_proj"),
        "model.layers.1.mlp.shared_experts.gate_up_proj": ("gate_proj", "up_proj"),
    }
    for destination, modules in fused_modules.items():
        prefix = destination.rsplit(".", 1)[0]
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}.{module}.{suffix}" for module in modules]
            if keys[0] in exported:
                exported[f"{destination}.{suffix}"] = mx.concatenate(
                    [exported.pop(key) for key in keys], axis=0
                )
    restored = target.sanitize(exported)
    assert restored.keys() == original.keys()
    for key in original:
        assert mx.array_equal(restored[key], original[key]).item(), key
    target.load_weights(list(restored.items()), strict=True)
    tokens = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])
    expected = source(tokens, cache=source.make_cache()).logits
    actual = target(tokens, cache=target.make_cache()).logits
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("prefill_step_size", [None, 4])
@pytest.mark.parametrize("second_prefix", [0, 32])
def test_glm_mixed_batch_preserves_state_through_padding(
    prefill_step_size, second_prefix
):
    from mlx_vlm.apc import APCManager, make_warm_batch_exact_cache_multi
    from mlx_vlm.generate.ar import PromptProcessingBatch

    mx.random.seed(31)
    model = tiny_glm()
    manager = APCManager(num_blocks=0, block_size=16)
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(model)
    prompts = [
        [i % 50 + 1 for i in range(47)],
        [i % 30 + 51 for i in range(second_prefix + 22)],
    ]
    source = model.make_cache()
    model(mx.array([prompts[0][:32]]), cache=source)
    second_source = model.make_cache()
    if second_prefix:
        model(mx.array([prompts[1][:second_prefix]]), cache=second_source)
    caches, _ = make_warm_batch_exact_cache_multi(
        [source, second_source], [32, second_prefix]
    )
    # Exact capacity makes finalize roll real padding into the leading positions;
    # spare zero-filled capacity must not be required for correct indexer validity.
    for cache in caches[1].caches:
        cache.step = 1
    suffixes = [prompts[0][32:], prompts[1][second_prefix:]]
    padded = mx.array([suffixes[0] + [0] * 7, suffixes[1]])
    batch = PromptProcessingBatch(
        model=model,
        uids=[0, 1],
        input_ids=suffixes,
        max_tokens=[1, 1],
        inputs_embeds=model.model.embed_tokens(padded),
        prompt_kwargs={},
        warm_cache=caches,
        prefill_step_size=prefill_step_size,
        right_pad_per_row=[7, 0],
        suffix_lens=[15, 22],
        apc_manager=manager,
        apc_coordinator=coordinator,
        apc_meta=[
            {
                "full_input_ids": tokens,
                "prefix_len": prefix,
                "checkpoint_lengths": coordinator.checkpoint_lengths(tokens, set()),
            }
            for tokens, prefix in zip(prompts, [32, second_prefix])
        ],
    )
    while batch.needs_processing():
        assert batch.prompt_step() > 0
    batch.generate(lambda lp: mx.argmax(lp, axis=-1), [lambda _: False] * 2)
    references = []
    for row, tokens in enumerate(prompts):
        reference = model.make_cache()
        for start in range(0, len(tokens), 4):
            model(mx.array([tokens[start : start + 4]]), cache=reference)
        for state in (0, 1):
            assert mx.allclose(
                caches[0][state][row : row + 1],
                reference[0][state],
                atol=1e-4,
                rtol=1e-4,
            ).item(), (row, state)
        references.append(reference)
    # Force the same next tokens to compare logits independently of sampling.
    actual = model(mx.array([[11], [12]]), cache=caches).logits
    for row, reference in enumerate(references):
        expected = model(mx.array([[11 + row]]), cache=reference).logits
        assert mx.allclose(actual[row : row + 1], expected, atol=1e-4, rtol=1e-4).item()
