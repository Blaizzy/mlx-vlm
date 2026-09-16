"""Fresh tiny language configurations shared by speculative and gradient tests."""

from mlx_vlm.models.deepseek_v4.config import ModelConfig as DeepseekConfig
from mlx_vlm.models.glm5_next.config import TextConfig as GlmConfig
from mlx_vlm.models.qwen3_5.config import TextConfig as QwenConfig
from mlx_vlm.models.qwen3_5_moe.config import TextConfig as QwenMoeConfig


def _text_dimensions(**overrides):
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        **overrides,
    }


def _qwen_config(config_type, **overrides):
    values = _text_dimensions(
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-06,
        head_dim=8,
        full_attention_interval=1,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    return config_type(**(values | overrides))


def tiny_qwen_text_config():
    return _qwen_config(
        QwenConfig,
        model_type="qwen3_5_text",
        intermediate_size=32,
        tie_word_embeddings=True,
    )


def tiny_qwen_moe_text_config(num_experts=4, moe_intermediate_size=8):
    return _qwen_config(
        QwenMoeConfig,
        model_type="qwen3_5_moe_text",
        num_experts=num_experts,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=moe_intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
    )


def tiny_deepseek_config():
    return DeepseekConfig(
        **_text_dimensions(
            intermediate_size=32,
            moe_intermediate_size=4,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            q_lora_rank=8,
            qk_rope_head_dim=4,
            head_dim=8,
            o_groups=1,
            o_lora_rank=8,
            index_n_heads=1,
            index_head_dim=8,
            index_topk=1,
            num_hash_layers=0,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            compress_ratios=[0],
            sliding_window=16,
        )
    )


def tiny_glm_text_config():
    return GlmConfig(
        **_text_dimensions(
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            kv_lora_rank=4,
            q_lora_rank=8,
            qk_nope_head_dim=4,
            v_head_dim=4,
            mlp_layer_types=["dense", "sparse"],
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            indexer_types=["full", "full"],
            index_topk=4,
            index_kpool=2,
            index_head_dim=4,
            index_n_heads=2,
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
            },
            hc_mult=2,
            max_position_embeddings=64,
        )
    )
