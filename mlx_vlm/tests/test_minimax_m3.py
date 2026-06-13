import math
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU
from PIL import Image
from transformers import BaseImageProcessor

import mlx_vlm.models.minimax_m3_vl.language as minimax_language
import mlx_vlm.utils as utils
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.generate.dispatch import (
    _drop_unused_multimodal_inputs,
    _trim_prompt_cache_entry_to,
)
from mlx_vlm.models.minimax_m3_vl.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.minimax_m3_vl.language import (
    LanguageModel,
    MiniMaxAttention,
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
    MiniMaxMLP,
    MiniMaxPackedSwitchGLU,
    MiniMaxSparseMoeBlock,
    MiniMaxSwiGLUOAI,
    _swiglu_oai,
)
from mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import Model
from mlx_vlm.models.minimax_m3_vl.processing_minimax_m3_vl import (
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLProcessor,
    MiniMaxM3VLVideoProcessor,
)
from mlx_vlm.models.minimax_m3_vl.vision import (
    MiniMaxVisionTransformer,
    _apply_vision_rope,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.eagle3 import _eagle3_verify_target
from mlx_vlm.utils import (
    _minimax_m3_frame_indices,
    _patch_minimax_m3_resize_processor,
    _video_metadata_from_array,
    prepare_inputs,
)
from mlx_vlm.video_generate import _video_inputs_to_generation_args


def _tiny_minimax_text_config(num_hidden_layers=2, **kwargs):
    return TextConfig(
        hidden_size=8,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=num_hidden_layers,
        rotary_dim=4,
        vocab_size=32,
        moe_layer_freq=[0] * num_hidden_layers,
        **kwargs,
    )


def test_minimax_m3_swiglu_oai_uses_configurable_beta():
    up = mx.array([[-2.0, 0.5, 3.0]], dtype=mx.float32)
    gate = mx.array([[-3.0, 0.25, 5.0]], dtype=mx.float32)

    out = _swiglu_oai(up, gate, alpha=1.7, limit=1.5, beta=0.25)
    clipped_gate = mx.clip(gate, a_min=None, a_max=1.5)
    clipped_up = mx.clip(up, a_min=-1.5, a_max=1.5)
    expected = clipped_gate * mx.sigmoid(1.7 * clipped_gate) * (clipped_up + 0.25)

    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6)


def _quantized_linear(input_dims, output_dims):
    linear = nn.Linear(input_dims, output_dims, bias=False)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    quantized.scales = quantized.scales.astype(mx.bfloat16)
    quantized.biases = quantized.biases.astype(mx.bfloat16)
    return quantized


def test_minimax_m3_fused_quantized_decode_linears_match_separate_outputs():
    linears = (
        _quantized_linear(64, 32),
        _quantized_linear(64, 16),
        _quantized_linear(64, 24),
    )
    x = mx.arange(64, dtype=mx.bfloat16).reshape(1, 1, 64) / 64

    separate = tuple(linear(x) for linear in linears)
    fused = minimax_language._decode_quantized_linears_fused(linears, x)
    mx.eval(*separate, *fused)

    assert fused is not None
    for actual, expected in zip(fused, separate):
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            rtol=1e-5,
        )


def test_minimax_m3_decode_projection_fusion_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_DISABLE_DECODE_FUSION", "1")
    linears = (_quantized_linear(64, 32), _quantized_linear(64, 16))
    x = mx.ones((1, 1, 64), dtype=mx.bfloat16)

    fused = minimax_language._decode_quantized_linears_fused(linears, x)

    assert fused is None


def test_minimax_m3_sparse_decode_fuses_main_and_index_projections(monkeypatch):
    config = _tiny_minimax_text_config(
        num_hidden_layers=1,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)
    calls = []

    def fake_fused(linears, x):
        calls.append(len(linears))
        if len(linears) != 5:
            return None
        B, L, _ = x.shape
        return (
            mx.zeros((B, L, config.num_attention_heads * config.head_dim)),
            mx.zeros((B, L, config.num_key_value_heads * config.head_dim)),
            mx.zeros((B, L, config.num_key_value_heads * config.head_dim)),
            mx.zeros((B, L, 4)),
            mx.zeros((B, L, 4)),
        )

    monkeypatch.setattr(minimax_language, "_decode_quantized_linears_fused", fake_fused)

    out = attention(mx.ones((1, 1, config.hidden_size), dtype=mx.float32))
    mx.eval(out)

    assert calls == [5]


def test_minimax_m3_swiglu_beta_reaches_dense_and_moe_paths():
    dense_config = _tiny_minimax_text_config(num_hidden_layers=1)
    dense_config.swiglu_beta = 0.5
    dense_lm = LanguageModel(dense_config)

    moe_config = _tiny_minimax_text_config(num_hidden_layers=1)
    moe_config.moe_layer_freq = [1]
    moe_config.swiglu_beta = 0.25
    moe_block = MiniMaxSparseMoeBlock(moe_config)

    assert dense_lm.model.layers[0].mlp.act_fn.beta == 0.5
    assert moe_block.switch_mlp.activation.beta == 0.25
    assert moe_block.pack_shared_expert is True
    assert moe_block.shared_experts is None


def test_minimax_m3_moe_respects_optional_shared_experts_and_routing_bias():
    config = _tiny_minimax_text_config(num_hidden_layers=1)
    config.moe_layer_freq = [1]
    config.n_shared_experts = 0
    config.use_routing_bias = False
    block = MiniMaxSparseMoeBlock(config)

    out = block(mx.ones((1, 2, config.hidden_size)))
    mx.eval(out)

    assert block.shared_experts is None
    assert block.e_score_correction_bias is None
    assert out.shape == (1, 2, config.hidden_size)


def test_minimax_m3_packed_switch_glu_matches_routed_plus_shared_mlp():
    hidden_size = 4
    intermediate_size = 3
    num_experts = 2
    activation = MiniMaxSwiGLUOAI(alpha=1.3, limit=4.0, beta=0.2)
    split = SwitchGLU(
        hidden_size,
        intermediate_size,
        num_experts,
        activation=activation,
    )
    shared = MiniMaxMLP(
        hidden_size,
        intermediate_size,
        alpha=1.3,
        limit=4.0,
        beta=0.2,
    )
    packed = MiniMaxPackedSwitchGLU(
        hidden_size,
        intermediate_size,
        num_experts + 1,
        activation=MiniMaxSwiGLUOAI(alpha=1.3, limit=4.0, beta=0.2),
    )

    def values(shape, offset):
        return (mx.arange(math.prod(shape), dtype=mx.float32).reshape(shape) + offset) / 50

    split.gate_proj.weight = values((num_experts, intermediate_size, hidden_size), 1)
    split.up_proj.weight = values((num_experts, intermediate_size, hidden_size), 17)
    split.down_proj.weight = values((num_experts, hidden_size, intermediate_size), 31)
    shared.gate_proj.weight = values((intermediate_size, hidden_size), 47)
    shared.up_proj.weight = values((intermediate_size, hidden_size), 59)
    shared.down_proj.weight = values((hidden_size, intermediate_size), 71)

    packed.gate_up_proj.weight = mx.concatenate(
        [
            mx.concatenate([split.gate_proj.weight, split.up_proj.weight], axis=1),
            mx.expand_dims(
                mx.concatenate(
                    [shared.gate_proj.weight, shared.up_proj.weight], axis=0
                ),
                axis=0,
            ),
        ],
        axis=0,
    )
    packed.down_proj.weight = mx.concatenate(
        [split.down_proj.weight, mx.expand_dims(shared.down_proj.weight, axis=0)],
        axis=0,
    )

    x = values((1, 24, hidden_size), 3)
    inds = mx.array([[[i % 2, (i + 1) % 2] for i in range(24)]], dtype=mx.int32)
    scores = mx.array([[[0.65, 0.35] for _ in range(24)]], dtype=mx.float32)
    expected = (split(x, inds) * scores[..., None]).sum(axis=-2) + shared(x)

    shared_inds = mx.full((1, 24, 1), num_experts, dtype=mx.int32)
    packed_inds = mx.concatenate([inds, shared_inds], axis=-1)
    packed_scores = mx.concatenate(
        [scores, mx.ones((1, 24, 1), dtype=mx.float32)], axis=-1
    )
    actual = (packed(x, packed_inds) * packed_scores[..., None]).sum(axis=-2)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5)


def test_minimax_m3_default_layer_frequency_matches_m3_shape():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=5,
        rotary_dim=4,
        vocab_size=32,
    )

    assert config.moe_layer_freq == [0, 0, 0, 1, 1]
    assert config.sparse_attention_config["sparse_attention_freq"] == [
        0,
        0,
        0,
        1,
        1,
    ]
    assert config.sparse_attention_config["sparse_disable_index_value"] == [
        0,
        0,
        0,
        1,
        1,
    ]
    assert config.sparse_attention_config["sparse_block_size"] == 128
    assert config.sparse_attention_config["sparse_topk_blocks"] == 16
    assert config.sparse_attention_config["sparse_local_block"] == 1
    assert config.is_moe_layer(2) is False
    assert config.is_moe_layer(3) is True
    assert config.has_sparse_index(2) is False
    assert config.has_sparse_index(3) is True


def test_minimax_m3_rotary_dim_defaults_from_partial_rotary_factor():
    config = TextConfig(
        hidden_size=24,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.25,
        num_hidden_layers=1,
        vocab_size=32,
    )

    assert config.rotary_dim == 2


def test_minimax_m3_explicit_rotary_dim_overrides_partial_rotary_factor():
    config = TextConfig(
        hidden_size=24,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=8,
        rotary_dim=6,
        partial_rotary_factor=0.25,
        num_hidden_layers=1,
        vocab_size=32,
    )

    assert config.rotary_dim == 6


def test_minimax_m3_config_accepts_transformers_layer_type_aliases():
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "num_hidden_layers": 4,
                "mlp_layer_types": ["dense", "sparse", "dense", "sparse"],
                "layer_types": [
                    "full_attention",
                    "minimax_m3_sparse",
                    "full_attention",
                    "minimax_m3_sparse",
                ],
                "index_n_heads": 2,
                "index_head_dim": 16,
                "index_block_size": 64,
                "index_topk_blocks": 8,
                "index_local_blocks": 0,
            },
            "vision_config": {},
        }
    ).text_config

    assert config.moe_layer_freq == [0, 1, 0, 1]
    assert config.sparse_attention_config["sparse_attention_freq"] == [
        0,
        1,
        0,
        1,
    ]
    assert config.sparse_attention_config["sparse_disable_index_value"] == [
        0,
        1,
        0,
        1,
    ]
    assert config.sparse_attention_config["sparse_num_index_heads"] == 2
    assert config.sparse_attention_config["sparse_index_dim"] == 16
    assert config.sparse_attention_config["sparse_block_size"] == 64
    assert config.sparse_attention_config["sparse_topk_blocks"] == 8
    assert config.sparse_attention_config["sparse_local_block"] == 0
    assert config.is_moe_layer(1) is True
    assert config.is_moe_layer(2) is False
    assert config.has_sparse_index(1) is True
    assert config.has_sparse_index(2) is False


def test_minimax_m3_layer_types_enable_empty_sparse_attention_config():
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "minimax_m3_sparse"],
                "sparse_attention_config": {},
                "index_n_heads": 2,
            },
            "vision_config": {},
        }
    ).text_config

    assert config.sparse_attention_config["sparse_attention_freq"] == [0, 1]
    assert config.sparse_attention_config["use_sparse_attention"] is True
    assert config.sparse_attention_config["sparse_disable_index_value"] == [0, 1]
    assert config.sparse_attention_config["sparse_num_index_heads"] == 2
    assert config.has_sparse_index(0) is False
    assert config.has_sparse_index(1) is True


def test_minimax_m3_sparse_disable_index_value_falls_back_to_attention_frequency():
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "num_hidden_layers": 4,
                "sparse_attention_config": {
                    "sparse_disable_index_value": [0, 1, 0, 1],
                    "sparse_num_index_heads": 2,
                },
            },
            "vision_config": {},
        }
    ).text_config

    assert config.sparse_attention_config["sparse_attention_freq"] == [0, 1, 0, 1]
    assert config.sparse_attention_config["sparse_disable_index_value"] == [
        0,
        1,
        0,
        1,
    ]
    assert config.sparse_attention_config["use_sparse_attention"] is True
    assert config.has_sparse_index(0) is False
    assert config.has_sparse_index(1) is True
    assert config.has_sparse_index(2) is False
    assert config.has_sparse_index(3) is True


def test_minimax_m3_sparse_disable_index_value_respects_disabled_sparse_attention():
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "num_hidden_layers": 2,
                "sparse_attention_config": {
                    "use_sparse_attention": False,
                    "sparse_disable_index_value": [1, 1],
                },
            },
            "vision_config": {},
        }
    ).text_config

    assert config.sparse_attention_config["sparse_attention_freq"] == [1, 1]
    assert config.sparse_attention_config["use_sparse_attention"] is False
    assert config.has_sparse_index(0) is False
    assert config.has_sparse_index(1) is False


def test_minimax_m3_attention_forwards_rope_scaling(monkeypatch):
    captured = {}

    class Rope:
        pass

    def fake_initialize_rope(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Rope()

    monkeypatch.setattr(minimax_language, "initialize_rope", fake_initialize_rope)
    config = _tiny_minimax_text_config(
        num_hidden_layers=1,
        max_position_embeddings=64,
        rope_scaling={"rope_type": "yarn", "factor": 2.0},
    )

    attention = MiniMaxAttention(config, layer_idx=0)

    assert isinstance(attention.rope, Rope)
    assert config.rope_scaling == {"rope_type": "yarn", "factor": 2.0, "type": "yarn"}
    assert captured["args"] == (config.rotary_dim, config.rope_theta)
    assert captured["kwargs"] == {
        "traditional": False,
        "scaling_config": config.rope_scaling,
        "max_position_embeddings": 64,
    }


def test_minimax_m3_attention_uses_position_ids_for_rope_offsets(monkeypatch):
    config = _tiny_minimax_text_config(
        num_hidden_layers=1,
        sparse_attention_config={"use_sparse_attention": False},
    )
    attention = MiniMaxAttention(config, layer_idx=0)
    offsets = []

    class Rope:
        def __call__(self, x, offset=0):
            offsets.append(offset)
            return x

    def fake_sdpa(queries, keys, values, cache, scale, mask):
        del keys, values, cache, scale, mask
        return mx.zeros_like(queries)

    attention.rope = Rope()
    monkeypatch.setattr(minimax_language, "scaled_dot_product_attention", fake_sdpa)

    output = attention(
        mx.ones((2, 1, config.hidden_size), dtype=mx.float32),
        position_ids=mx.array([[4], [7]], dtype=mx.int32),
    )

    mx.eval(output)
    assert [offset.tolist() for offset in offsets] == [[4, 7], [4, 7]]


def test_minimax_m3_config_uses_top_level_compression_as_vision_fallback():
    config = ModelConfig.from_dict(
        {
            "text_config": {},
            "vision_config": {},
            "img_token_compression_config": {
                "spatial_merge_size": 4,
                "temporal_patch_size": 3,
            },
        }
    )

    assert config.img_token_compression_config == {
        "spatial_merge_size": 4,
        "temporal_patch_size": 3,
    }
    assert config.vision_config.img_token_compression_config == {
        "spatial_merge_size": 4,
        "temporal_patch_size": 3,
    }
    assert config.vision_config.spatial_merge_size == 4
    assert config.vision_config.temporal_patch_size == 3


def test_minimax_m3_config_keeps_explicit_vision_compression():
    config = ModelConfig.from_dict(
        {
            "text_config": {},
            "vision_config": {
                "img_token_compression_config": {
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                },
            },
            "img_token_compression_config": {
                "spatial_merge_size": 4,
                "temporal_patch_size": 3,
            },
        }
    )

    assert config.vision_config.img_token_compression_config == {
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.temporal_patch_size == 2


def test_minimax_m3_config_coerces_direct_dict_subconfigs():
    config = ModelConfig(
        text_config={
            "num_hidden_layers": 2,
            "mlp_layer_types": ["dense", "sparse"],
        },
        vision_config={
            "img_token_compression_config": {
                "spatial_merge_size": 3,
                "temporal_patch_size": 1,
            },
        },
    )

    assert isinstance(config.text_config, TextConfig)
    assert isinstance(config.vision_config, VisionConfig)
    assert config.text_config.moe_layer_freq == [0, 1]
    assert config.vision_config.spatial_merge_size == 3
    assert config.vision_config.temporal_patch_size == 1


def test_minimax_m3_direct_config_uses_top_level_compression_fallback():
    config = ModelConfig(
        text_config={},
        vision_config={},
        img_token_compression_config={
            "spatial_merge_size": 4,
            "temporal_patch_size": 3,
        },
    )

    assert isinstance(config.text_config, TextConfig)
    assert isinstance(config.vision_config, VisionConfig)
    assert config.vision_config.img_token_compression_config == {
        "spatial_merge_size": 4,
        "temporal_patch_size": 3,
    }
    assert config.vision_config.spatial_merge_size == 4
    assert config.vision_config.temporal_patch_size == 3


def test_minimax_m3_config_preserves_top_level_checkpoint_metadata():
    config = ModelConfig.from_dict(
        {
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "auto_map": {
                "AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig",
            },
            "eos_token_id": 200020,
            "image_grid_pinpoints": "[(336, 336), (336, 672)]",
            "num_reward_heads": 0,
            "text_config": {},
            "transformers_version": "4.52.4",
            "vision_config": {},
        }
    )

    round_tripped = config.to_dict()

    assert round_tripped["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"]
    assert round_tripped["auto_map"] == {
        "AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig",
    }
    assert round_tripped["eos_token_id"] == 200020
    assert round_tripped["image_grid_pinpoints"] == "[(336, 336), (336, 672)]"
    assert round_tripped["num_reward_heads"] == 0
    assert round_tripped["transformers_version"] == "4.52.4"


def test_minimax_m3_text_config_preserves_top_level_causal_lm_fields():
    from mlx_vlm.models.minimax_m3 import ModelConfig as TextOnlyMiniMaxM3Config

    config = TextOnlyMiniMaxM3Config.from_dict(
        {
            "model_type": "minimax_m3",
            "hidden_size": 32,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "text_config": {},
        }
    )

    assert config.model_type == "minimax_m3"
    assert config.hidden_size == 32
    assert config.num_hidden_layers == 2
    assert config.vocab_size == 128


def test_minimax_m3_text_config_sanitizes_quantization_keys():
    from mlx_vlm.models.minimax_m3 import ModelConfig as TextOnlyMiniMaxM3Config

    quantization = {
        "bits": 4,
        "group_size": 64,
        "model.layers.3.block_sparse_moe.gate": {"bits": 8, "group_size": 64},
        "lm_head": False,
        "ignored_layers": ["model.embed_tokens", "lm_head"],
    }
    config = TextOnlyMiniMaxM3Config.from_dict(
        {
            "model_type": "minimax_m3",
            "num_hidden_layers": 4,
            "quantization": quantization,
            "quantization_config": quantization,
        }
    )

    assert "model.layers.3.block_sparse_moe.gate" not in config.quantization
    assert config.quantization[
        "language_model.model.layers.3.block_sparse_moe.gate"
    ] == {
        "bits": 8,
        "group_size": 64,
    }
    assert config.quantization["language_model.lm_head"] is False
    assert config.quantization["ignored_layers"] == [
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    ]
    assert config.quantization_config is config.quantization


def test_minimax_m3_text_model_sanitize_maps_causal_lm_checkpoint_keys():
    from mlx_vlm.models.minimax_m3 import Model as TextOnlyMiniMaxM3Model
    from mlx_vlm.models.minimax_m3 import ModelConfig as TextOnlyMiniMaxM3Config

    config = TextOnlyMiniMaxM3Config(
        hidden_size=8,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        moe_layer_freq=[1],
        num_local_experts=2,
        n_shared_experts=0,
    )
    model = TextOnlyMiniMaxM3Model(config)

    weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.arange(16, dtype=mx.uint8).reshape(
            2, 8
        ),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": mx.ones(
            (2, 2), dtype=mx.uint8
        ),
    }
    prefix = "model.layers.0.block_sparse_moe"
    for expert in range(config.num_local_experts):
        for old_name in ("w1", "w2", "w3"):
            key = f"{prefix}.experts.{expert}.{old_name}"
            weights[f"{key}.weight"] = mx.full((2, 8), expert + 1, dtype=mx.uint8)
            weights[f"{key}.weight_scale_inv"] = mx.full(
                (2, 2), expert + 2, dtype=mx.uint8
            )

    out = model.sanitize(weights)
    prefix = "language_model.model.layers.0.block_sparse_moe"

    q_weight = out["language_model.model.layers.0.self_attn.q_proj.weight"]
    assert q_weight.dtype == mx.uint32
    assert q_weight.shape == (2, 2)
    assert "language_model.model.layers.0.self_attn.q_proj.scales" in out

    gate_weight = out[f"{prefix}.switch_mlp.gate_proj.weight"]
    gate_scales = out[f"{prefix}.switch_mlp.gate_proj.scales"]
    assert gate_weight.dtype == mx.uint32
    assert gate_weight.shape == (2, 2, 2)
    assert gate_scales.shape == (2, 2, 2)
    assert f"{prefix}.experts.0.w1.weight" not in out
    assert f"{prefix}.experts.0.w1.scales" not in out


def test_minimax_m3_text_model_sanitize_packs_hf_shared_expert_layout():
    from mlx_vlm.models.minimax_m3 import Model as TextOnlyMiniMaxM3Model
    from mlx_vlm.models.minimax_m3 import ModelConfig as TextOnlyMiniMaxM3Config

    config = TextOnlyMiniMaxM3Config(
        hidden_size=8,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        moe_layer_freq=[1],
        num_local_experts=2,
        n_shared_experts=1,
    )
    model = TextOnlyMiniMaxM3Model(config)

    source_prefix = "model.layers.0.block_sparse_moe"
    weights = {}
    for suffix in ("weight", "scales", "biases"):
        for expert in range(config.num_local_experts):
            weights[f"{source_prefix}.experts.{expert}.w1.{suffix}"] = mx.full(
                (2, 3), expert + 1, dtype=mx.float32
            )
            weights[f"{source_prefix}.experts.{expert}.w3.{suffix}"] = mx.full(
                (2, 3), expert + 3, dtype=mx.float32
            )
            weights[f"{source_prefix}.experts.{expert}.w2.{suffix}"] = mx.full(
                (4, 2), expert + 5, dtype=mx.float32
            )
        weights[f"{source_prefix}.shared_experts.gate_proj.{suffix}"] = mx.full(
            (2, 3), 7, dtype=mx.float32
        )
        weights[f"{source_prefix}.shared_experts.up_proj.{suffix}"] = mx.full(
            (2, 3), 8, dtype=mx.float32
        )
        weights[f"{source_prefix}.shared_experts.down_proj.{suffix}"] = mx.full(
            (4, 2), 9, dtype=mx.float32
        )

    out = model.sanitize(weights)
    prefix = "language_model.model.layers.0.block_sparse_moe"
    gate_up = out[f"{prefix}.switch_mlp.gate_up_proj.weight"]
    down = out[f"{prefix}.switch_mlp.down_proj.weight"]
    mx.eval(gate_up, down)

    assert gate_up.shape == (3, 4, 3)
    assert down.shape == (3, 4, 2)
    assert np.all(np.array(gate_up[0, :2]) == 1)
    assert np.all(np.array(gate_up[0, 2:]) == 3)
    assert np.all(np.array(gate_up[1, :2]) == 2)
    assert np.all(np.array(gate_up[1, 2:]) == 4)
    assert np.all(np.array(gate_up[2, :2]) == 7)
    assert np.all(np.array(gate_up[2, 2:]) == 8)
    assert np.all(np.array(down[0]) == 5)
    assert np.all(np.array(down[1]) == 6)
    assert np.all(np.array(down[2]) == 9)
    assert out[f"{prefix}.switch_mlp.gate_up_proj.scales"].shape == (3, 4, 3)
    assert out[f"{prefix}.switch_mlp.gate_up_proj.biases"].shape == (3, 4, 3)
    assert f"{prefix}.experts.0.w1.weight" not in out
    assert f"{prefix}.experts.0.w2.weight" not in out
    assert f"{prefix}.experts.0.w3.weight" not in out
    assert f"{prefix}.shared_experts.gate_proj.weight" not in out


def test_minimax_m3_config_preserves_quantization_metadata_with_aliases():
    quantization = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "model.layers.3.block_sparse_moe.gate": {"bits": 8, "group_size": 64},
        "lm_head": False,
        "ignored_layers": [
            "model.embed_tokens",
            "lm_head",
            "model.language_model.model.layers.3.self_attn.q_proj",
            "model.vision_tower",
            "vision_tower",
        ],
    }
    config = ModelConfig.from_dict(
        {
            "text_config": {},
            "vision_config": {},
            "quantization": quantization,
            "quantization_config": quantization,
        }
    )

    assert "model.layers.3.block_sparse_moe.gate" not in config.quantization
    assert config.quantization[
        "language_model.model.layers.3.block_sparse_moe.gate"
    ] == {
        "bits": 8,
        "group_size": 64,
    }
    assert config.quantization["language_model.lm_head"] is False
    assert config.quantization["ignored_layers"] == [
        "language_model.model.embed_tokens",
        "language_model.lm_head",
        "language_model.model.layers.3.self_attn.q_proj",
        "vision_tower",
        "vision_tower",
    ]
    assert config.quantization_config is config.quantization


def test_minimax_m3_sparse_index_requires_explicit_frequency():
    config = _tiny_minimax_text_config(num_hidden_layers=1)
    config.moe_layer_freq = [1]
    config.sparse_attention_config = {"use_sparse_attention": True}

    lm = LanguageModel(config)

    assert lm.layers[0].is_moe_layer is True
    assert lm.layers[0].self_attn.has_sparse_index is False
    assert not hasattr(lm.layers[0].self_attn, "index_q_proj")


def test_minimax_m3_router_quantization_override_is_explicit_affine():
    lm = LanguageModel(_tiny_minimax_text_config(num_hidden_layers=1))

    assert lm.quant_predicate("layers.0.block_sparse_moe.gate", None) == {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
    }


def test_minimax_m3_prompt_template_inserts_visual_tokens():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        "Describe the inputs.",
        num_images=2,
        video=["clip-a.mp4", "clip-b.mp4"],
    )

    assert prompt == ("]<]image[>[" * 2 + "]<]video[>[" * 2 + "Describe the inputs.")


def test_minimax_m3_prompt_template_targets_last_user_turn():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        [
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "What is in it?"},
        ],
        num_images=1,
    )

    assert prompt == (
        "User: Remember this.\n"
        "Assistant: Done.\n"
        "User: ]<]image[>[What is in it?\n"
        "Assistant:"
    )


def test_minimax_m3_video_prompt_template_targets_last_user_turn():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        [
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "What happens next?"},
        ],
        video=["clip-a.mp4", "clip-b.mp4"],
    )

    assert prompt == (
        "User: Remember this.\n"
        "Assistant: Done.\n"
        "User: ]<]video[>[]<]video[>[What happens next?\n"
        "Assistant:"
    )


def test_minimax_m3_prompt_template_uses_video_count_without_refs():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        "Describe the inputs.",
        num_videos=2,
    )

    assert prompt == "]<]video[>[" * 2 + "Describe the inputs."


def test_minimax_m3_video_count_targets_last_user_turn():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        [
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "What happens next?"},
        ],
        num_videos=2,
    )

    assert prompt == (
        "User: Remember this.\n"
        "Assistant: Done.\n"
        "User: ]<]video[>[]<]video[>[What happens next?\n"
        "Assistant:"
    )


def test_minimax_m3_prompt_template_counts_typed_media_content():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "image-a.png"},
                {"type": "video_url", "video_url": {"url": "video-a.mp4"}},
                {"type": "text", "text": "Describe both."},
            ],
        },
    )

    assert prompt == "]<]image[>[" + "]<]video[>[" + "Describe both."


def test_minimax_m3_typed_media_count_targets_last_user_turn():
    config = {"model_type": "minimax_m3_vl"}

    prompt = apply_chat_template(
        None,
        config,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "image-a.png"},
                    {"type": "text", "text": "Remember this."},
                ],
            },
            {"role": "assistant", "content": "Done."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "video-a.mp4"},
                    {"type": "text", "text": "What changed?"},
                ],
            },
        ],
    )

    assert prompt == (
        "User: Remember this.\n"
        "Assistant: Done.\n"
        "User: ]<]video[>[What changed?\n"
        "Assistant:"
    )


def test_minimax_m3_enable_thinking_maps_to_template_mode():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = kwargs
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        "Think.",
        enable_thinking=True,
    )

    assert prompt == "prompt"
    assert processor.kwargs["thinking_mode"] == "enabled"


def test_minimax_m3_text_only_enable_thinking_maps_to_template_mode():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = kwargs
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3"},
        "Think.",
        enable_thinking=True,
    )

    assert prompt == "prompt"
    assert processor.kwargs["thinking_mode"] == "enabled"


def test_minimax_m3_disable_thinking_maps_to_template_mode():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = kwargs
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        "Answer directly.",
        enable_thinking=False,
    )

    assert prompt == "prompt"
    assert processor.kwargs["thinking_mode"] == "disabled"


def test_minimax_m3_explicit_thinking_mode_overrides_enable_flag():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = kwargs
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        "Think if useful.",
        enable_thinking=False,
        thinking_mode="adaptive",
    )

    assert prompt == "prompt"
    assert processor.kwargs["thinking_mode"] == "adaptive"


def test_minimax_m3_omitted_thinking_flag_uses_template_default():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            self.kwargs = kwargs
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        "Use the default.",
    )

    assert prompt == "prompt"
    assert "thinking_mode" not in processor.kwargs


def test_minimax_m3_chat_forwards_thinking_mode_to_template_only(monkeypatch):
    import mlx_vlm.chat as chat_mod

    class Model:
        config = {"model_type": "minimax_m3_vl"}

    class Processor:
        pass

    class Chunk:
        text = "ok"

    captured_template = {}
    captured_stream = {}

    def fake_apply_chat_template(*args, **kwargs):
        del args
        captured_template.update(kwargs)
        return "prompt"

    def fake_stream_generate(*args, **kwargs):
        del args
        captured_stream.update(kwargs)
        yield Chunk()

    monkeypatch.setattr(chat_mod, "load", lambda model_path: (Model(), Processor()))
    monkeypatch.setattr(chat_mod.MLXVisionChat, "print_help", lambda self: None)
    monkeypatch.setattr(chat_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(chat_mod, "stream_generate", fake_stream_generate)

    session = chat_mod.MLXVisionChat(
        model_path="m3",
        enable_thinking=False,
        thinking_mode="adaptive",
    )
    session.add_to_history("user", "Think when needed.")

    assert session.generate_response() == "ok"
    assert captured_template["enable_thinking"] is False
    assert captured_template["thinking_mode"] == "adaptive"
    assert "thinking_mode" not in captured_stream


def test_minimax_m3_chat_keeps_default_thinking_mode_adaptive(monkeypatch):
    import mlx_vlm.chat as chat_mod

    class Model:
        config = {"model_type": "minimax_m3_vl"}

    class Processor:
        pass

    class Chunk:
        text = "ok"

    captured_template = {}

    def fake_apply_chat_template(*args, **kwargs):
        del args
        captured_template.update(kwargs)
        return "prompt"

    def fake_stream_generate(*args, **kwargs):
        del args, kwargs
        yield Chunk()

    monkeypatch.setattr(chat_mod, "load", lambda model_path: (Model(), Processor()))
    monkeypatch.setattr(chat_mod.MLXVisionChat, "print_help", lambda self: None)
    monkeypatch.setattr(chat_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(chat_mod, "stream_generate", fake_stream_generate)

    session = chat_mod.MLXVisionChat(model_path="m3")
    session.add_to_history("user", "Use the model default.")

    assert session.generate_response() == "ok"
    assert "enable_thinking" not in captured_template
    assert "thinking_mode" not in captured_template


def test_minimax_m3_template_decodes_top_level_tool_arguments():
    class Processor:
        chat_template = "template"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            self.messages = messages
            return "prompt"

    processor = Processor()

    prompt = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "lookup",
                        "arguments": '{"param-1":"value-1","skip":null}',
                    },
                    {
                        "function": {
                            "name": "save",
                            "arguments": '{"items":["a","b"]}',
                        }
                    },
                ],
            }
        ],
    )

    assert prompt == "prompt"
    calls = processor.messages[0]["tool_calls"]
    assert calls[0]["arguments"] == {"param-1": "value-1", "skip": None}
    assert calls[1]["function"]["arguments"] == {"items": ["a", "b"]}


def test_minimax_m3_prepare_inputs_uses_native_processor_for_images():
    class MiniMaxM3VLImageProcessor(BaseImageProcessor):
        def preprocess(self, images=None, **kwargs):
            raise AssertionError("generic image preprocessing should be skipped")

    class MiniMaxVLProcessor:
        def __init__(self):
            self.image_processor = MiniMaxM3VLImageProcessor()
            self.video_processor = None
            self.tokenizer = type(
                "Tokenizer", (), {"pad_token": "pad", "eos_token": "eos"}
            )()
            self.text = None
            self.images = None

        def __call__(
            self,
            text=None,
            images=None,
            videos=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            del videos, padding, return_tensors, kwargs
            self.text = text
            self.images = images
            return {
                "input_ids": np.array([[1, 2]], dtype=np.int32),
                "attention_mask": np.array([[1, 1]], dtype=np.int32),
                "pixel_values": np.zeros((1, 4), dtype=np.float32),
                "image_grid_thw": np.array([[1, 1, 1]], dtype=np.int32),
            }

    processor = MiniMaxVLProcessor()
    image = Image.new("RGB", (16, 16))

    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts="]<]image[>[Describe.",
    )

    assert processor.text == "]<]image[>[Describe."
    assert len(processor.images) == 1
    assert processor.images[0] is image
    assert inputs["input_ids"].tolist() == [[1, 2]]
    assert inputs["pixel_values"].shape == (1, 4)


def test_minimax_m3_prepare_inputs_uses_native_processor_for_video_with_image_processor():
    class MiniMaxM3VLImageProcessor(BaseImageProcessor):
        def preprocess(self, images=None, **kwargs):
            raise AssertionError("generic image preprocessing should be skipped")

    class MiniMaxM3VLVideoProcessor:
        pass

    class MiniMaxVLProcessor:
        def __init__(self):
            self.image_processor = MiniMaxM3VLImageProcessor()
            self.video_processor = MiniMaxM3VLVideoProcessor()
            self.tokenizer = type(
                "Tokenizer", (), {"pad_token": "pad", "eos_token": "eos"}
            )()
            self.kwargs = None
            self.videos = None

        def __call__(
            self,
            text=None,
            images=None,
            videos=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            del text, images, padding, return_tensors
            self.kwargs = kwargs
            self.videos = videos
            return {
                "input_ids": np.array([[1]], dtype=np.int32),
                "attention_mask": np.array([[1]], dtype=np.int32),
                "pixel_values_videos": np.zeros((4, 8), dtype=np.float32),
                "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    processor = MiniMaxVLProcessor()
    video = np.zeros((2, 3, 24, 24), dtype=np.uint8)

    inputs = prepare_inputs(
        processor,
        videos=[video],
        prompts="]<]video[>[Describe.",
        fps=2.0,
    )

    assert len(processor.videos) == 1
    assert processor.videos[0] is video
    assert processor.kwargs["do_resize"] is True
    assert processor.kwargs["fps"] == 2.0
    assert processor.kwargs["video_metadata"][0]["frames_indices"] == [0, 1]
    assert inputs["pixel_values_videos"].shape == (4, 8)


def test_minimax_m3_prepare_inputs_resizes_raw_videos_by_default():
    class MiniMaxM3VLVideoProcessor:
        pass

    class MiniMaxVLProcessor:
        def __init__(self):
            self.video_processor = MiniMaxM3VLVideoProcessor()
            self.do_resize = None

        def __call__(
            self,
            text=None,
            images=None,
            videos=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            self.do_resize = kwargs.get("do_resize")
            return {
                "input_ids": np.array([[1]], dtype=np.int32),
                "attention_mask": np.array([[1]], dtype=np.int32),
                "pixel_values_videos": np.zeros((4, 8), dtype=np.float32),
                "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    processor = MiniMaxVLProcessor()

    prepare_inputs(
        processor,
        videos=[np.zeros((2, 3, 37, 53), dtype=np.uint8)],
        prompts="]<]video[>[Describe.",
    )

    assert processor.do_resize is True


def test_minimax_m3_prepare_inputs_forwards_video_metadata(monkeypatch):
    class MiniMaxM3VLVideoProcessor:
        pass

    class MiniMaxVLProcessor:
        def __init__(self):
            self.video_processor = MiniMaxM3VLVideoProcessor()
            self.kwargs = None

        def __call__(
            self,
            text=None,
            images=None,
            videos=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            self.kwargs = kwargs
            return {
                "input_ids": np.array([[1]], dtype=np.int32),
                "attention_mask": np.array([[1]], dtype=np.int32),
                "pixel_values_videos": np.zeros((4, 8), dtype=np.float32),
                "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    metadata = {
        "total_num_frames": 12,
        "fps": 6.0,
        "width": 80,
        "height": 40,
        "duration": 2.0,
        "video_backend": "opencv",
        "frames_indices": [0, 3, 6, 9],
    }

    def fake_load_video(*args, **kwargs):
        assert kwargs["return_metadata"] is True
        assert kwargs["sampling_strategy"] == "minimax_m3"
        return np.zeros((4, 3, 40, 80), dtype=np.uint8), 2.0, metadata

    monkeypatch.setattr(utils, "load_video", fake_load_video)
    processor = MiniMaxVLProcessor()

    prepare_inputs(
        processor,
        videos=["clip.mp4"],
        prompts="]<]video[>[Describe.",
        fps=2.0,
    )

    assert processor.kwargs["do_resize"] is True
    assert processor.kwargs["fps"] == 2.0
    assert processor.kwargs["video_metadata"] == [metadata]


def test_minimax_m3_prepare_inputs_accepts_explicit_video_metadata():
    class MiniMaxM3VLVideoProcessor:
        pass

    class MiniMaxVLProcessor:
        def __init__(self):
            self.video_processor = MiniMaxM3VLVideoProcessor()
            self.kwargs = None

        def __call__(
            self,
            text=None,
            images=None,
            videos=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            self.kwargs = kwargs
            return {
                "input_ids": np.array([[1]], dtype=np.int32),
                "attention_mask": np.array([[1]], dtype=np.int32),
                "pixel_values_videos": np.zeros((4, 8), dtype=np.float32),
                "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            }

    metadata = {
        "total_num_frames": 6,
        "fps": 3.0,
        "width": 40,
        "height": 20,
        "duration": 2.0,
        "video_backend": "custom",
        "frames_indices": [0, 2, 4],
    }
    processor = MiniMaxVLProcessor()

    prepare_inputs(
        processor,
        videos=[np.zeros((3, 3, 20, 40), dtype=np.uint8)],
        prompts="]<]video[>[Describe.",
        fps=1.5,
        video_metadata=[metadata],
    )

    assert processor.kwargs["do_resize"] is True
    assert processor.kwargs["fps"] == 1.5
    assert processor.kwargs["video_metadata"] == [metadata]


class _MiniMaxFakeTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0

    vocab = {
        MiniMaxM3VLProcessor.IMAGE_TOKEN: 11,
        MiniMaxM3VLProcessor.VIDEO_TOKEN: 12,
        MiniMaxM3VLProcessor.VISION_START_TOKEN: 13,
        MiniMaxM3VLProcessor.VISION_END_TOKEN: 14,
    }

    def __init__(self):
        self.seen_text = None

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(
        self,
        text,
        padding=True,
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        del add_special_tokens, return_tensors, kwargs
        self.seen_text = list(text)
        encoded = [self._encode(prompt) for prompt in text]
        max_len = max(len(ids) for ids in encoded)
        if padding:
            encoded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        attention_mask = [
            [0 if token == self.pad_token_id else 1 for token in ids]
            for ids in encoded
        ]
        return {"input_ids": encoded, "attention_mask": attention_mask}

    def _encode(self, text):
        specials = sorted(self.vocab, key=len, reverse=True)
        ids = []
        index = 0
        while index < len(text):
            for special in specials:
                if text.startswith(special, index):
                    ids.append(self.vocab[special])
                    index += len(special)
                    break
            else:
                ids.append(9)
                index += 1
        return ids


def _tiny_minimax_processor():
    return MiniMaxM3VLProcessor(
        image_processor=MiniMaxM3VLImageProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=16,
            do_resize=False,
        ),
        video_processor=MiniMaxM3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=16,
            do_resize=False,
        ),
        tokenizer=_MiniMaxFakeTokenizer(),
    )


def test_minimax_m3_processor_expands_image_tokens_and_patchifies():
    processor = _tiny_minimax_processor()
    image = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)

    outputs = processor(
        text=f"{MiniMaxM3VLProcessor.IMAGE_TOKEN}Describe.",
        images=[image],
        return_tensors=None,
        return_mm_token_type_ids=True,
    )

    expected_text = (
        MiniMaxM3VLProcessor.VISION_START_TOKEN
        + MiniMaxM3VLProcessor.IMAGE_TOKEN
        + MiniMaxM3VLProcessor.VISION_END_TOKEN
        + "Describe."
    )
    assert processor.tokenizer.seen_text == [expected_text]
    assert outputs["pixel_values"].shape == (4, 24)
    assert outputs["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert outputs["input_ids"][0].count(processor.image_token_id) == 1
    image_pos = outputs["input_ids"][0].index(processor.image_token_id)
    assert outputs["mm_token_type_ids"][0][image_pos] == 1


def test_minimax_m3_processor_expands_video_tokens_with_timestamps():
    processor = _tiny_minimax_processor()
    video = np.arange(3 * 3 * 4 * 4, dtype=np.uint8).reshape(3, 3, 4, 4)
    metadata = {"fps": 2.0, "frames_indices": [0, 1, 2]}

    outputs = processor(
        text=f"{MiniMaxM3VLProcessor.VIDEO_TOKEN}Describe.",
        videos=[video],
        video_metadata=[metadata],
        return_tensors=None,
        return_mm_token_type_ids=True,
    )

    frame_chunk = (
        MiniMaxM3VLProcessor.VISION_START_TOKEN
        + MiniMaxM3VLProcessor.VIDEO_TOKEN
        + MiniMaxM3VLProcessor.VISION_END_TOKEN
    )
    expected_text = f"]<]0.0 seconds[>[{frame_chunk}]<]1.0 seconds[>[{frame_chunk}Describe."
    assert processor.tokenizer.seen_text == [expected_text]
    assert outputs["pixel_values_videos"].shape == (8, 24)
    assert outputs["video_grid_thw"].tolist() == [[2, 2, 2]]
    assert outputs["input_ids"][0].count(processor.video_token_id) == 2
    for index, token_id in enumerate(outputs["input_ids"][0]):
        if token_id == processor.video_token_id:
            assert outputs["mm_token_type_ids"][0][index] == 2


def _legacy_smart_resize(
    height, width, factor=28, min_pixels=4 * 28 * 28, max_pixels=451584
):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def test_minimax_m3_resize_patch_adds_long_side_image_resize(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__], "smart_resize", _legacy_smart_resize, raising=False
    )

    class MiniMaxM3VLImageProcessor:
        patch_size = 14
        merge_size = 2
        max_pixels = 451584
        valid_kwargs = type(
            "MiniMaxImageKwargs",
            (),
            {"__annotations__": {"return_tensors": object, "max_pixels": int}},
        )

        def preprocess(self, images, do_resize=True, return_tensors=None, **kwargs):
            height, width = images[0]
            if do_resize:
                height, width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    max_pixels=self.max_pixels,
                )
            return {
                "image_grid_thw": np.array(
                    [[1, height // self.patch_size, width // self.patch_size]],
                    dtype=np.int32,
                )
            }

        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            height, width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                max_pixels=self.max_pixels,
            )
            return height // self.patch_size * (width // self.patch_size)

    processor = type("MiniMaxVLProcessor", (), {})()
    processor.image_processor = MiniMaxM3VLImageProcessor()
    processor.video_processor = None

    _patch_minimax_m3_resize_processor(processor)

    assert (
        "max_long_side_pixel" in processor.image_processor.valid_kwargs.__annotations__
    )
    assert (
        processor.image_processor.get_number_of_image_patches(
            2048, 1024, {"max_long_side_pixel": 1008}
        )
        == 72 * 36
    )
    assert processor.image_processor.preprocess(
        [(2048, 1024)], max_long_side_pixel=1008
    )["image_grid_thw"].tolist() == [[1, 72, 36]]
    assert processor.image_processor.preprocess([(200, 40)], max_long_side_pixel=1008)[
        "image_grid_thw"
    ].tolist() == [[1, 40, 8]]


def test_minimax_m3_resize_patch_adds_long_side_video_resize_and_cap(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__], "smart_resize", _legacy_smart_resize, raising=False
    )

    class FakeVideo:
        def __init__(self, shape):
            self.shape = shape

    class MiniMaxM3VLVideoProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        max_pixels = 602112
        min_pixels = 4 * 28 * 28
        valid_kwargs = type(
            "MiniMaxVideoKwargs",
            (),
            {"__annotations__": {"return_tensors": object, "max_pixels": int}},
        )

        def preprocess(
            self, videos=None, do_resize=True, return_tensors=None, **kwargs
        ):
            frames, _, height, width = videos[0].shape
            if do_resize:
                height, width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
            return {
                "video_grid_thw": np.array(
                    [
                        [
                            math.ceil(frames / self.temporal_patch_size),
                            height // self.patch_size,
                            width // self.patch_size,
                        ]
                    ],
                    dtype=np.int32,
                )
            }

    processor = type("MiniMaxVLProcessor", (), {})()
    processor.image_processor = None
    processor.video_processor = MiniMaxM3VLVideoProcessor()

    _patch_minimax_m3_resize_processor(processor)

    assert (
        "max_long_side_pixel" in processor.video_processor.valid_kwargs.__annotations__
    )
    assert processor.video_processor.preprocess(
        videos=[FakeVideo((2, 3, 2048, 1024))],
        max_long_side_pixel=1008,
    )["video_grid_thw"].tolist() == [[1, 72, 36]]
    with np.testing.assert_raises_regex(ValueError, "max_total_pixels"):
        processor.video_processor.preprocess(
            videos=[FakeVideo((400, 3, 2048, 2048))],
            max_long_side_pixel=1008,
        )


def test_minimax_m3_frame_sampling_matches_timestamp_rule():
    assert _minimax_m3_frame_indices(
        total_frames=10,
        video_fps=10.0,
        fps=2.0,
    ) == [0, 5, 9]
    assert _minimax_m3_frame_indices(
        total_frames=10,
        video_fps=10.0,
        fps=3.0,
    ) == [0, 4, 8, 9]
    assert _minimax_m3_frame_indices(
        total_frames=1,
        video_fps=10.0,
        fps=2.0,
    ) == [0]


def test_minimax_m3_video_array_metadata_uses_fps_for_timestamps():
    metadata = _video_metadata_from_array(np.zeros((4, 3, 40, 80), dtype=np.uint8), 2.0)

    assert metadata["total_num_frames"] == 4
    assert metadata["fps"] == 2.0
    assert metadata["frames_indices"] == [0, 1, 2, 3]
    assert metadata["height"] == 40
    assert metadata["width"] == 80


def test_minimax_m3_video_generate_keeps_video_pixels_in_kwargs():
    inputs = {
        "input_ids": mx.array([[1, 2, 3]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
        "pixel_values_videos": mx.ones((2, 4), dtype=mx.float32),
        "video_grid_thw": mx.array([[1, 1, 2]], dtype=mx.int32),
    }

    input_ids, pixel_values, mask, kwargs = _video_inputs_to_generation_args(
        inputs, "minimax_m3_vl"
    )

    assert input_ids.tolist() == [[1, 2, 3]]
    assert mask.tolist() == [[1, 1, 1]]
    assert pixel_values is None
    assert kwargs["pixel_values_videos"].shape == (2, 4)
    assert kwargs["video_grid_thw"].tolist() == [[1, 1, 2]]


def test_video_generate_non_minimax_keeps_legacy_video_pixel_mapping():
    inputs = {
        "input_ids": mx.array([[1, 2, 3]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
        "pixel_values_videos": mx.ones((2, 4), dtype=mx.float32),
        "video_grid_thw": mx.array([[1, 1, 2]], dtype=mx.int32),
    }

    _, pixel_values, _, kwargs = _video_inputs_to_generation_args(inputs, "qwen2_5_vl")

    assert pixel_values.shape == (2, 4)
    assert "pixel_values_videos" not in kwargs
    assert kwargs["video_grid_thw"].tolist() == [[1, 1, 2]]


def test_prompt_cache_trim_keeps_minimax_m3_index_cache_in_sync():
    cache = MiniMaxM3KVCache()
    keys = mx.ones((1, 1, 5, 4), dtype=mx.float32)
    values = mx.ones((1, 1, 5, 4), dtype=mx.float32)
    index_keys = mx.ones((1, 1, 5, 4), dtype=mx.float32)

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(index_keys)
    _trim_prompt_cache_entry_to(cache, 3)

    assert cache.offset == 3
    assert cache.index_offset == 3
    assert cache.state[0][0].shape[2] == 3
    assert cache.state[1].shape[2] == 3


def test_minimax_m3_empty_cache_state_roundtrips():
    cache = MiniMaxM3KVCache()

    state = cache.state
    restored = MiniMaxM3KVCache()
    restored.state = state

    assert state == (None, None)
    assert restored.empty() is True
    assert restored.offset == 0
    assert restored.index_offset == 0
    assert restored.index_keys is None


def test_minimax_m3_empty_batch_cache_state_preserves_padding():
    cache = MiniMaxM3BatchKVCache([2, 0])

    state = cache.state
    restored = MiniMaxM3BatchKVCache([0])
    restored.state = state
    mx.eval(state[0][2], state[0][3])

    assert state[0][0] is None
    assert state[0][1] is None
    assert state[1] is None
    assert restored.empty() is True
    assert restored.offset.tolist() == [-2, 0]
    assert restored.left_padding.tolist() == [2, 0]
    assert restored.index_offset == 0
    assert restored.index_keys is None


def test_minimax_m3_to_batch_preserves_warm_kv_and_index_cache():
    cache = MiniMaxM3KVCache()
    keys = mx.arange(12, dtype=mx.float32).reshape(1, 1, 3, 4)
    values = mx.arange(15, dtype=mx.float32).reshape(1, 1, 3, 5)
    index_keys = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(index_keys)
    batch = cache.to_batch([2])
    extracted = batch.extract(0)

    assert batch.kv_cache.keys.shape == (1, 1, 5, 4)
    assert batch.kv_cache.values.shape == (1, 1, 5, 5)
    assert batch.kv_cache.offset.tolist() == [3]
    assert batch.left_padding.tolist() == [2]
    assert batch.index_offset == 5
    assert batch.index_keys[:, :, :2, :].sum().item() == 0
    assert batch.index_keys[:, :, 2:, :].tolist() == index_keys.tolist()
    assert extracted.kv_cache.state[0].tolist() == keys.tolist()
    assert extracted.kv_cache.state[1].tolist() == values.tolist()
    assert extracted.index_keys.tolist() == index_keys.tolist()


def test_minimax_m3_eagle3_verifier_captures_hidden_states():
    lm = LanguageModel(_tiny_minimax_text_config(num_hidden_layers=2))
    prompt_cache = lm.make_cache()
    verify_input = mx.array([[1, 2, 3]], dtype=mx.int32)

    def sampler(logits):
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    hidden, target_tokens, gdn_states = _eagle3_verify_target(
        lm,
        verify_input,
        prompt_cache=prompt_cache,
        sampler=sampler,
        target_layer_ids=[0, 1],
    )

    assert hidden.shape == (1, 3, 16)
    assert target_tokens.shape == (1, 3)
    assert gdn_states is None


def test_minimax_m3_rollback_speculative_cache_trims_index_cache():
    lm = LanguageModel(_tiny_minimax_text_config(num_hidden_layers=1))
    cache = MiniMaxM3KVCache()
    keys = mx.ones((1, 1, 5, 4), dtype=mx.float32)
    values = mx.ones((1, 1, 5, 4), dtype=mx.float32)
    index_keys = mx.ones((1, 1, 5, 4), dtype=mx.float32)

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(index_keys)
    accepted = lm.rollback_speculative_cache([cache], None, accepted=1, block_size=4)

    assert accepted == 1
    assert cache.offset == 3
    assert cache.index_offset == 3


def test_minimax_m3_batch_rollback_zeroes_rejected_index_tails():
    lm = LanguageModel(_tiny_minimax_text_config(num_hidden_layers=1))
    cache = MiniMaxM3BatchKVCache([0, 0])
    keys = mx.ones((2, 1, 5, 4), dtype=mx.float32)
    values = mx.ones((2, 1, 5, 4), dtype=mx.float32)
    index_keys = mx.ones((2, 1, 5, 4), dtype=mx.float32)

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(index_keys)
    accepted = lm.rollback_speculative_cache(
        [cache], None, accepted=mx.array([2, 0], dtype=mx.int32), block_size=4
    )

    assert accepted == 2
    assert cache._idx == 4
    assert cache.index_offset == 4
    assert cache.kv_cache.keys[0, :, 1:4, :].sum().item() > 0
    assert cache.index_keys[0, :, 1:4, :].sum().item() > 0
    assert cache.kv_cache.keys[1, :, 2:4, :].sum().item() == 0
    assert cache.kv_cache.values[1, :, 2:4, :].sum().item() == 0
    assert cache.index_keys[1, :, 2:4, :].sum().item() == 0


def test_multimodal_prefix_trim_drops_unused_minimax_video_kwargs():
    class ModelStub:
        config = type(
            "Config",
            (),
            {"image_token_index": 5, "video_token_index": 6},
        )()

    pixel_values = mx.ones((1, 2), dtype=mx.float32)
    kwargs = {
        "cached_image_features": mx.ones((1, 2), dtype=mx.float32),
        "cached_video_features": mx.ones((3, 2), dtype=mx.float32),
        "pixel_values_videos": mx.ones((3, 2), dtype=mx.float32),
        "video_grid_thw": mx.array([[1, 1, 3]], dtype=mx.int32),
    }

    pixel_values = _drop_unused_multimodal_inputs(
        ModelStub(), [1, 2, 3], pixel_values, kwargs
    )

    assert pixel_values is None
    assert "cached_image_features" not in kwargs
    assert "cached_video_features" not in kwargs
    assert "pixel_values_videos" not in kwargs
    assert "video_grid_thw" not in kwargs


def test_multimodal_prefix_trim_keeps_minimax_video_kwargs_for_video_suffix():
    class ModelStub:
        config = type(
            "Config",
            (),
            {"image_token_index": 5, "video_token_index": 6},
        )()

    pixel_values = mx.ones((1, 2), dtype=mx.float32)
    kwargs = {
        "cached_image_features": mx.ones((1, 2), dtype=mx.float32),
        "cached_video_features": mx.ones((3, 2), dtype=mx.float32),
        "pixel_values_videos": mx.ones((3, 2), dtype=mx.float32),
        "video_grid_thw": mx.array([[1, 1, 3]], dtype=mx.int32),
    }

    pixel_values = _drop_unused_multimodal_inputs(
        ModelStub(), [6, 42], pixel_values, kwargs
    )

    assert pixel_values is None
    assert "cached_image_features" not in kwargs
    assert "cached_video_features" in kwargs
    assert kwargs["pixel_values_videos"].shape == (3, 2)
    assert kwargs["video_grid_thw"].tolist() == [[1, 1, 3]]


def test_multimodal_prefix_trim_keeps_image_pixels_for_image_suffix():
    class ModelStub:
        config = type(
            "Config",
            (),
            {"image_token_index": 5, "video_token_index": 6},
        )()

    pixel_values = mx.ones((1, 2), dtype=mx.float32)
    kwargs = {
        "cached_image_features": mx.ones((1, 2), dtype=mx.float32),
        "cached_video_features": mx.ones((3, 2), dtype=mx.float32),
        "pixel_values_videos": mx.ones((3, 2), dtype=mx.float32),
        "video_grid_thw": mx.array([[1, 1, 3]], dtype=mx.int32),
    }

    result = _drop_unused_multimodal_inputs(ModelStub(), [5, 42], pixel_values, kwargs)

    assert result is pixel_values
    assert "cached_image_features" in kwargs
    assert "cached_video_features" not in kwargs
    assert "pixel_values_videos" not in kwargs
    assert "video_grid_thw" not in kwargs


def test_minimax_m3_batch_index_cache_filter_extend_extract():
    cache = MiniMaxM3BatchKVCache([1, 0])
    keys = mx.ones((2, 1, 3, 4))
    values = mx.ones((2, 1, 3, 4))

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(keys)
    cache.filter(mx.array([0]))

    assert cache.index_offset == 2
    assert cache.state[1].shape == (1, 1, 2, 4)
    assert cache.left_padding.tolist() == [0]

    extracted = cache.extract(0)
    assert extracted.index_keys.shape == (1, 1, 2, 4)

    cache_a = MiniMaxM3BatchKVCache([0])
    cache_a.update_and_fetch(mx.ones((1, 1, 2, 4)), mx.ones((1, 1, 2, 4)))
    cache_a.update_index_and_fetch(mx.ones((1, 1, 2, 4)))

    cache_b = MiniMaxM3BatchKVCache([0])
    cache_b.update_and_fetch(mx.ones((1, 1, 4, 4)), mx.ones((1, 1, 4, 4)))
    cache_b.update_index_and_fetch(mx.ones((1, 1, 4, 4)))

    cache_a.extend(cache_b)
    assert cache_a.index_offset == 4
    assert cache_a.state[1].shape == (2, 1, 4, 4)


def test_minimax_m3_kv_cache_merge_returns_batch_cache():
    caches = []
    for offset in (1, 2):
        cache = MiniMaxM3KVCache()
        cache.update_and_fetch(
            mx.ones((1, 1, 3, 4), dtype=mx.float32) * offset,
            mx.ones((1, 1, 3, 5), dtype=mx.float32) * offset,
        )
        cache.update_index_and_fetch(
            mx.ones((1, 2, 3, 4), dtype=mx.float32) * offset
        )
        caches.append(cache)

    merged = MiniMaxM3KVCache.merge(caches)
    extracted = merged.extract(1)

    assert isinstance(merged, MiniMaxM3BatchKVCache)
    assert merged.kv_cache.keys.shape == (2, 1, 3, 4)
    assert merged.kv_cache.values.shape == (2, 1, 3, 5)
    assert merged.index_keys.shape == (2, 2, 3, 4)
    assert extracted.kv_cache.state[0].tolist() == caches[1].kv_cache.state[0].tolist()
    assert extracted.index_keys.tolist() == caches[1].state[1].tolist()


def test_minimax_m3_batch_cache_merge_preserves_index_cache_rows():
    warm = MiniMaxM3KVCache()
    warm.update_and_fetch(
        mx.ones((1, 1, 3, 4), dtype=mx.float32),
        mx.ones((1, 1, 3, 5), dtype=mx.float32),
    )
    warm.update_index_and_fetch(mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4))
    cold = MiniMaxM3KVCache()

    merged = MiniMaxM3BatchKVCache.merge([warm, cold])

    assert merged.kv_cache.keys.shape == (2, 1, 3, 4)
    assert merged.kv_cache.values.shape == (2, 1, 3, 5)
    assert merged.index_offset == 3
    assert merged.index_keys.shape == (2, 2, 3, 4)
    assert merged.left_padding.tolist() == [0, 3]
    assert merged.offset.tolist() == [3, 0]
    assert merged.index_keys[0].tolist() == warm.index_keys[0, :, :3, :].tolist()
    assert merged.index_keys[1].sum().item() == 0


def test_minimax_m3_batch_cache_preserves_sparse_index_with_kv_quantization():
    config = _tiny_minimax_text_config(num_hidden_layers=1)
    config.sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_attention_freq": [1],
        "sparse_index_dim": 4,
        "sparse_num_index_heads": 1,
        "sparse_block_size": 2,
        "sparse_topk_blocks": 1,
    }
    lm = LanguageModel(config)

    batch_cache = _make_cache(lm, [0], kv_bits=4, kv_group_size=64)
    sparse_cache = batch_cache[0]

    assert isinstance(sparse_cache, MiniMaxM3BatchKVCache)
    assert hasattr(sparse_cache, "update_index_and_fetch")
    index_keys = sparse_cache.update_index_and_fetch(mx.ones((1, 1, 2, 4)))
    assert index_keys.shape == (1, 1, 2, 4)


def test_minimax_m3_batch_forced_msa_forward(monkeypatch):
    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_FORCE_MSA", "1")
    config = TextConfig(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=32,
        shared_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)
    cache = MiniMaxM3BatchKVCache([0, 1])

    output = attention(mx.ones((2, 3, 16)), cache=cache)
    mx.eval(output, cache.state)

    assert output.shape == (2, 3, 16)
    assert cache.state[1].shape == (2, 1, 3, 4)


def test_minimax_m3_shard_partitions_sparse_index_queries(monkeypatch):
    class Group:
        def size(self):
            return 2

    calls = []

    def fake_shard_linear(module, sharding, *, group=None, segments=1):
        calls.append((module, sharding, group, segments))
        return module

    def fake_shard_inplace(module, sharding, *, group=None, segments=1):
        calls.append((module, sharding, group, segments))

    config = TextConfig(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=32,
        shared_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
        },
    )
    lm = LanguageModel(config)
    attention = lm.layers[0].self_attn
    index_q_proj = attention.index_q_proj
    index_k_proj = attention.index_k_proj

    monkeypatch.setattr(minimax_language, "shard_linear", fake_shard_linear)
    monkeypatch.setattr(minimax_language, "shard_inplace", fake_shard_inplace)

    lm.shard(Group())

    assert attention.num_attention_heads == 2
    assert attention.num_key_value_heads == 1
    assert attention.index_heads == 1
    assert any(
        module is index_q_proj and sharding == "all-to-sharded"
        for module, sharding, _, _ in calls
    )
    assert not any(module is index_k_proj for module, _, _, _ in calls)


def test_minimax_m3_shard_rejects_unshardable_sparse_index_heads(monkeypatch):
    class Group:
        def size(self):
            return 2

    def fake_shard_linear(module, sharding, *, group=None, segments=1):
        return module

    config = TextConfig(
        hidden_size=12,
        intermediate_size=8,
        dense_intermediate_size=24,
        shared_intermediate_size=8,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 3,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
        },
    )
    lm = LanguageModel(config)
    monkeypatch.setattr(minimax_language, "shard_linear", fake_shard_linear)

    with np.testing.assert_raises_regex(ValueError, "sparse index heads"):
        lm.shard(Group())


def test_minimax_m3_sparse_selection_respects_attention_mask():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 1,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array([[[[1.0]]]], dtype=mx.float32)
    idx_keys = mx.array([[[[100.0], [100.0], [1.0], [1.0]]]], dtype=mx.float32)
    mask = mx.array([[[[False, False, True, True]]]])

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 3, mask)

    assert sparse_mask.tolist() == [[[[False, False, True, True]]]]


def test_minimax_m3_sparse_selection_merges_2d_causal_mask_on_query_axis():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 1,
            "sparse_topk_blocks": 3,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.ones((1, 1, 3, 1), dtype=mx.float32)
    idx_keys = mx.ones((1, 1, 3, 1), dtype=mx.float32)
    mask = mx.array(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 0, mask)

    assert sparse_mask.shape == (1, 1, 3, 3)
    assert sparse_mask.tolist() == [
        [
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        ]
    ]


def test_minimax_m3_sparse_selection_merges_2d_padding_mask_on_batch_axis():
    sparse_mask = mx.ones((2, 1, 3, 4), dtype=mx.bool_)
    mask = mx.array(
        [
            [True, False, True, False],
            [False, True, True, False],
        ]
    )

    merged = MiniMaxAttention._merge_sparse_mask(sparse_mask, mask)

    assert merged.shape == (2, 1, 3, 4)
    assert merged[:, :, 0, :].tolist() == [
        [[True, False, True, False]],
        [[False, True, True, False]],
    ]
    assert merged[:, :, 2, :].tolist() == [
        [[True, False, True, False]],
        [[False, True, True, False]],
    ]


def test_minimax_m3_sparse_selection_prefers_batch_axis_for_ambiguous_2d_mask():
    sparse_mask = mx.ones((2, 1, 2, 3), dtype=mx.bool_)
    mask = mx.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ]
    )

    merged = MiniMaxAttention._merge_sparse_mask(sparse_mask, mask)
    valid = MiniMaxAttention._selection_valid_mask(mask, B=2, H_idx=1, L=2, total_len=3)

    assert merged.shape == (2, 1, 2, 3)
    assert merged[:, :, 0, :].tolist() == [
        [[True, False, True]],
        [[False, True, True]],
    ]
    assert merged[:, :, 1, :].tolist() == [
        [[True, False, True]],
        [[False, True, True]],
    ]
    assert valid.shape == (2, 1, 2, 3)
    assert valid[:, :, 0, :].tolist() == [
        [[True, False, True]],
        [[False, True, True]],
    ]


def test_minimax_m3_dense_attention_combines_integer_padding_with_causal_mask(
    monkeypatch,
):
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=8,
        sparse_attention_config={"use_sparse_attention": False},
    )
    attention = MiniMaxAttention(config, 0)
    captured = {}

    def fake_sdpa(queries, keys, values, cache, scale, mask):
        del keys, values, cache, scale
        captured["mask"] = mask
        return mx.zeros_like(queries)

    monkeypatch.setattr(minimax_language, "scaled_dot_product_attention", fake_sdpa)

    output = attention(
        mx.ones((1, 3, 4), dtype=mx.float32),
        mask=mx.array([[1, 1, 0]], dtype=mx.int32),
    )
    mx.eval(output)

    assert captured["mask"].shape == (1, 1, 3, 3)
    assert captured["mask"].tolist() == [
        [
            [
                [True, False, False],
                [True, True, False],
                [True, True, False],
            ]
        ]
    ]


def test_minimax_m3_sparse_decode_attention_matches_dense_masked_attention():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    queries = (mx.arange(8).reshape(1, 4, 1, 2) / 10).astype(mx.float32)
    keys = (mx.arange(24).reshape(1, 2, 6, 2) / 10).astype(mx.float32)
    values = (mx.arange(24, 48).reshape(1, 2, 6, 2) / 10).astype(mx.float32)
    idx_queries = mx.array([[[[1.0]], [[-1.0]]]], dtype=mx.float32)
    idx_keys = mx.array([[[[3.0], [3.0], [1.0], [1.0], [-2.0], [-2.0]]]])

    sparse_mask, topk_idx, topk_valid = attention._build_sparse_mask(
        idx_queries, idx_keys, 5, None, return_block_indices=True
    )
    dense = minimax_language.scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=attention.scale,
        mask=sparse_mask,
    )
    compact = attention._sparse_decode_attention(
        queries, keys, values, topk_idx, topk_valid, 5
    )
    mx.eval(dense, compact)

    assert compact is not None
    np.testing.assert_allclose(np.array(compact), np.array(dense), rtol=1e-5)


def test_minimax_m3_sparse_decode_all_valid_fastpath_matches_generic():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    queries = (mx.arange(8).reshape(1, 4, 1, 2) / 10).astype(mx.float32)
    keys = (mx.arange(28).reshape(1, 2, 7, 2) / 10).astype(mx.float32)
    values = (mx.arange(28, 56).reshape(1, 2, 7, 2) / 10).astype(mx.float32)
    topk_idx = mx.array([[[[0, 3]], [[1, 3]]]], dtype=mx.int32)
    topk_valid = mx.ones(topk_idx.shape, dtype=mx.bool_)

    generic = attention._sparse_decode_attention(
        queries,
        keys,
        values,
        topk_idx,
        topk_valid,
        q_start=6,
        topk_all_valid=False,
    )
    fast = attention._sparse_decode_attention(
        queries,
        keys,
        values,
        topk_idx,
        topk_valid,
        q_start=6,
        topk_all_valid=True,
    )
    mx.eval(generic, fast)

    np.testing.assert_allclose(np.array(fast), np.array(generic), rtol=1e-5)


def test_minimax_m3_sparse_decode_offset_cache_is_not_registered_state():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    queries = mx.ones((1, 4, 1, 2), dtype=mx.float32)
    keys = mx.ones((1, 2, 6, 2), dtype=mx.float32)
    values = mx.ones((1, 2, 6, 2), dtype=mx.float32)
    topk_idx = mx.array([[[[2]], [[2]]]], dtype=mx.int32)
    topk_valid = mx.ones(topk_idx.shape, dtype=mx.bool_)

    output = attention._sparse_decode_attention(
        queries,
        keys,
        values,
        topk_idx,
        topk_valid,
        q_start=5,
        topk_all_valid=True,
    )
    mx.eval(output)

    assert hasattr(attention, "_minimax_m3_sparse_block_offsets_cache")
    assert "_minimax_m3_sparse_block_offsets_cache" not in attention
    assert "_minimax_m3_sparse_block_offsets_cache" not in attention.parameters()


def test_minimax_m3_decode_uses_compacted_sparse_kv(monkeypatch):
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    cache = MiniMaxM3KVCache()
    key_lengths = []

    def fake_sdpa(queries, keys, values, cache, scale, mask):
        del values, cache, scale, mask
        key_lengths.append(keys.shape[2])
        return mx.zeros_like(queries)

    monkeypatch.setattr(minimax_language, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_FORCE_MSA", "1")
    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_SPARSE_DECODE_MAX_DENSITY", "1.0")

    mx.eval(attention(mx.ones((1, 5, 4), dtype=mx.float32), cache=cache))
    mx.eval(attention(mx.ones((1, 1, 4), dtype=mx.float32), cache=cache))

    assert key_lengths[-2:] == [5, 2]


def test_minimax_m3_sparse_decode_compaction_respects_density_threshold(monkeypatch):
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    queries = mx.ones((1, 2, 1, 2), dtype=mx.float32)
    keys = mx.ones((1, 1, 3, 2), dtype=mx.float32)

    monkeypatch.delenv("MLX_VLM_MINIMAX_M3_SPARSE_DECODE_MAX_DENSITY", raising=False)
    assert attention._can_use_sparse_decode_attention(queries, keys, None) is False

    keys = mx.ones((1, 1, 6, 2), dtype=mx.float32)
    assert attention._can_use_sparse_decode_attention(queries, keys, None) is True

    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_SPARSE_DECODE_MAX_DENSITY", "0.25")
    assert attention._can_use_sparse_decode_attention(queries, keys, None) is False

    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_SPARSE_DECODE_MAX_DENSITY", "0.5")
    assert attention._can_use_sparse_decode_attention(queries, keys, None) is True


def test_minimax_m3_sparse_decode_index_fastpath_matches_generic_selection():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array(
        [[[[1.0, 0.25]], [[0.25, 1.0]]]],
        dtype=mx.float32,
    )
    idx_keys = mx.array(
        [
            [
                [
                    [10.0, 0.0],
                    [9.0, 0.0],
                    [0.0, 10.0],
                    [0.0, 9.0],
                    [3.0, 3.0],
                    [2.0, 2.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )

    generic = attention._build_sparse_mask(
        idx_queries,
        idx_keys,
        q_start=idx_keys.shape[2] - 1,
        return_block_indices=True,
        build_token_mask=False,
    )
    fast = attention._build_sparse_decode_indices(
        idx_queries,
        idx_keys,
        q_start=idx_keys.shape[2] - 1,
    )
    mx.eval(generic[1], generic[2], fast[0], fast[1])

    generic_idx = generic[1].tolist()
    fast_idx = fast[0].tolist()
    for generic_head, fast_head in zip(generic_idx[0], fast_idx[0]):
        assert sorted(generic_head[0]) == sorted(fast_head[0])
    assert fast[1].tolist() == generic[2].tolist()


def test_minimax_m3_sparse_decode_index_fastpath_can_be_disabled(monkeypatch):
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)

    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_DISABLE_DECODE_INDEX_FASTPATH", "1")
    out = attention._build_sparse_decode_indices(
        mx.ones((1, 1, 1, 2), dtype=mx.float32),
        mx.ones((1, 1, 4, 2), dtype=mx.float32),
        q_start=3,
    )

    assert out is None


def test_minimax_m3_sparse_selection_ignores_nan_block_scores():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array([[[[1.0]]]], dtype=mx.float32)
    idx_keys = mx.array(
        [[[[float("nan")], [float("nan")], [2.0], [2.0], [1.0], [1.0]]]],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 5, None)

    assert sparse_mask.tolist() == [[[[False, False, True, True, False, False]]]]


def test_minimax_m3_sparse_selection_forces_local_block_over_scored_blocks():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array([[[[1000.0]]]], dtype=mx.float32)
    idx_keys = mx.array(
        [[[[0.0], [0.0], [1000.0], [1000.0], [1.0], [1.0]]]],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 5, None)

    assert sparse_mask.tolist() == [[[[False, False, False, False, True, True]]]]


def test_minimax_m3_sparse_lse_selection_uses_attention_head_scale():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        rotary_dim=4,
        vocab_size=32,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
            "sparse_score_type": "lse",
        },
    )
    attention = MiniMaxAttention(config, layer_idx=0)
    idx_queries = mx.ones((1, 1, 1, 1), dtype=mx.float32)
    idx_keys = mx.array([[[[2.0], [-100.0], [1.2], [1.2]]]], dtype=mx.float32)

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 3, None)

    assert sparse_mask.tolist() == [[[[False, False, True, True]]]]


def test_minimax_m3_sparse_selection_prioritizes_init_before_local_blocks():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 1,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array([[[[1.0]]]], dtype=mx.float32)
    idx_keys = mx.array(
        [[[[0.0], [0.0], [1000.0], [1000.0]]]],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 3, None)

    assert sparse_mask.tolist() == [[[[True, True, False, False]]]]


def test_minimax_m3_sparse_selection_is_per_index_head_and_gqa_group():
    config = TextConfig(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        rotary_dim=1,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=mx.float32)
    idx_keys = mx.array(
        [[[[10.0, 0.0], [10.0, 0.0], [0.0, 20.0], [0.0, 20.0]]]],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 3, None)

    assert sparse_mask.tolist() == [
        [
            [[True, True, False, False]],
            [[True, True, False, False]],
            [[False, False, True, True]],
            [[False, False, True, True]],
        ]
    ]


def test_minimax_m3_compiled_sparse_selection_matches_generic_causal_path():
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[0.5, 1.0], [1.0, 0.5], [0.25, 1.0]],
            ]
        ],
        dtype=mx.float32,
    )
    idx_keys = mx.array(
        [[[[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0]]]],
        dtype=mx.float32,
    )
    q_start = 3
    qpos = mx.arange(q_start, q_start + idx_queries.shape[2])
    kpos = mx.arange(idx_keys.shape[2])
    causal_mask = (kpos[None, :] <= qpos[:, None])[None, None, :, :]

    compiled = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, None, return_block_indices=True
    )
    generic = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, causal_mask, return_block_indices=True
    )
    mx.eval(*compiled, *generic)

    assert compiled[0].tolist() == generic[0].tolist()
    assert compiled[2].tolist() == generic[2].tolist()
    compiled_blocks = np.sort(np.array(compiled[1].tolist()), axis=-1)
    generic_blocks = np.sort(np.array(generic[1].tolist()), axis=-1)
    np.testing.assert_array_equal(compiled_blocks, generic_blocks)


def test_minimax_m3_compiled_sparse_selection_can_be_disabled(monkeypatch):
    config = TextConfig(
        hidden_size=8,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=8,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    attention = MiniMaxAttention(config, 0)
    idx_queries = mx.array(
        [[[[2.0, 0.0]], [[0.0, 2.0]]]],
        dtype=mx.float32,
    )
    idx_keys = mx.array(
        [[[[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0]]]],
        dtype=mx.float32,
    )
    q_start = 5
    causal_mask = mx.ones((1, 1, 1, idx_keys.shape[2]), dtype=mx.bool_)
    expected = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, causal_mask, return_block_indices=True
    )

    def raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("compiled sparse prefill path should be disabled")

    monkeypatch.setenv("MLX_VLM_MINIMAX_M3_DISABLE_COMPILED_SPARSE_PREFILL", "1")
    monkeypatch.setattr(
        minimax_language, "_build_sparse_causal_mask_compiled", raise_if_called
    )
    actual = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, None, return_block_indices=True
    )
    mx.eval(*expected, *actual)

    assert actual[0].tolist() == expected[0].tolist()
    assert actual[2].tolist() == expected[2].tolist()
    actual_blocks = np.sort(np.array(actual[1].tolist()), axis=-1)
    expected_blocks = np.sort(np.array(expected[1].tolist()), axis=-1)
    np.testing.assert_array_equal(actual_blocks, expected_blocks)


def test_minimax_m3_compiled_sparse_selection_handles_batched_index_heads():
    config = TextConfig(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=2,
        num_hidden_layers=1,
        rotary_dim=2,
        vocab_size=16,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 4,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    attention = MiniMaxAttention(config, 0)
    rng = np.random.default_rng(17)
    idx_queries = mx.array(
        rng.normal(size=(2, 4, 3, 2)).astype(np.float32), dtype=mx.float32
    )
    idx_keys = mx.array(
        rng.normal(size=(2, 1, 6, 2)).astype(np.float32), dtype=mx.float32
    )
    q_start = 3
    qpos = mx.arange(q_start, q_start + idx_queries.shape[2])
    kpos = mx.arange(idx_keys.shape[2])
    causal_mask = (kpos[None, :] <= qpos[:, None])[None, None, :, :]

    compiled = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, None, return_block_indices=True
    )
    generic = attention._build_sparse_mask(
        idx_queries, idx_keys, q_start, causal_mask, return_block_indices=True
    )
    mx.eval(*compiled, *generic)

    assert compiled[0].tolist() == generic[0].tolist()
    assert compiled[2].tolist() == generic[2].tolist()
    compiled_blocks = np.sort(np.array(compiled[1].tolist()), axis=-1)
    generic_blocks = np.sort(np.array(generic[1].tolist()), axis=-1)
    np.testing.assert_array_equal(compiled_blocks, generic_blocks)


def test_minimax_m3_vision_rope_axis_split_and_frame_segments():
    config = VisionConfig(
        hidden_size=80,
        intermediate_size=160,
        num_attention_heads=1,
        num_hidden_layers=0,
        img_token_compression_config={
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
        vision_segment_max_frames=4,
    )
    vision = MiniMaxVisionTransformer(config)
    grid_thw = mx.array([[5, 2, 2]], dtype=mx.int32)

    assert vision._segment_grid_thw(grid_thw) == [(4, 2, 2), (1, 2, 2)]

    freqs = vision._rotary_pos_emb(grid_thw)
    assert freqs.shape == (20, 78)
    assert mx.allclose(freqs[0], mx.zeros((78,), dtype=freqs.dtype)).item()

    x = mx.arange(80, dtype=mx.float32).reshape(1, 1, 1, 80)
    rotated = _apply_vision_rope(x, mx.ones((1, 78), dtype=mx.float32))
    assert mx.allclose(rotated[..., 78:], x[..., 78:]).item()


def test_minimax_m3_vision_returns_hidden_states_when_requested():
    config = VisionConfig(
        hidden_size=12,
        intermediate_size=24,
        num_attention_heads=1,
        num_hidden_layers=1,
        num_channels=1,
        patch_size=1,
        img_token_compression_config={
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
        },
    )
    vision = MiniMaxVisionTransformer(config)

    out, hidden_states = vision(
        mx.ones((2, 1), dtype=mx.float32),
        mx.array([[2, 1, 1]], dtype=mx.int32),
        output_hidden_states=True,
    )

    assert out.shape == (2, 12)
    assert len(hidden_states) == 2
    assert hidden_states[0].shape == (2, 12)
    assert hidden_states[-1].shape == (2, 12)


def test_minimax_m3_model_call_honors_precomputed_input_embeddings():
    config = ModelConfig(
        text_config=_tiny_minimax_text_config(num_hidden_layers=0),
        vision_config=VisionConfig(
            hidden_size=2,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        projector_hidden_size=2,
    )
    model = Model(config)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    inputs_embeds = mx.full((1, 3, config.text_config.hidden_size), 0.25)

    out = model(input_ids, inputs_embeds=inputs_embeds)
    expected = model.language_model(input_ids, inputs_embeds=inputs_embeds)
    recomputed = model.language_model(input_ids)

    np.testing.assert_allclose(
        np.array(out.logits),
        np.array(expected.logits),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not mx.allclose(out.logits, recomputed.logits).item()


def test_minimax_m3_selects_configured_vision_feature_layers():
    config = ModelConfig(
        text_config=_tiny_minimax_text_config(num_hidden_layers=0),
        vision_config=VisionConfig(
            hidden_size=2,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        projector_hidden_size=2,
        vocab_size=32,
        vision_feature_layer=[0, 1],
        vision_feature_select_strategy="default",
    )
    model = Model(config)
    hidden_states = (
        mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32),
        mx.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=mx.float32),
    )

    selected = model._select_vision_features(hidden_states)

    assert selected.tolist() == [
        [3.0, 4.0, 30.0, 40.0],
        [5.0, 6.0, 50.0, 60.0],
    ]


def test_minimax_m3_merge_visual_features_handles_image_and_video_tokens():
    inputs_embeds = mx.zeros((1, 6, 3), dtype=mx.float32)
    input_ids = mx.array([[1, 5, 2, 6, 6, 3]], dtype=mx.int32)
    image_features = mx.array([[1.0, 2.0, 3.0]])
    video_features = mx.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    merged, visual_mask = Model.merge_input_ids_with_visual_features(
        inputs_embeds,
        input_ids,
        image_features=image_features,
        video_features=video_features,
        image_token_index=5,
        video_token_index=6,
    )

    assert visual_mask.tolist() == [[False, True, False, True, True, False]]
    assert merged[0, 1].tolist() == [1.0, 2.0, 3.0]
    assert merged[0, 3].tolist() == [4.0, 5.0, 6.0]
    assert merged[0, 4].tolist() == [7.0, 8.0, 9.0]


def test_minimax_m3_get_input_embeddings_handles_images_and_videos():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=16,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        image_token_index=5,
        video_token_index=6,
        projector_hidden_size=3,
        vocab_size=16,
    )
    model = Model(config)
    model._compute_visual_features = lambda pixel_values, grid_thw: pixel_values
    input_ids = mx.array([[1, 5, 2, 6, 6, 3]], dtype=mx.int32)

    output = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=mx.array([[1.0, 2.0, 3.0]]),
        image_grid_thw=mx.array([[1, 1, 1]], dtype=mx.int32),
        pixel_values_videos=mx.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        video_grid_thw=mx.array([[2, 1, 1]], dtype=mx.int32),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, False, True, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [1.0, 2.0, 3.0]
    assert output.inputs_embeds[0, 3].tolist() == [4.0, 5.0, 6.0]
    assert output.inputs_embeds[0, 4].tolist() == [7.0, 8.0, 9.0]


def test_minimax_m3_encode_image_and_video_forward_grid_metadata():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=16,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
        ),
        projector_hidden_size=3,
        vocab_size=16,
    )
    model = Model(config)
    captured = {}

    def fake_compute_visual_features(pixel_values, grid_thw):
        captured["pixel_values"] = pixel_values
        captured["grid_thw"] = grid_thw
        return mx.ones((1, 3), dtype=mx.float32)

    pixel_values = mx.ones((1, 3), dtype=mx.float32)
    image_grid_thw = mx.array([[1, 1, 1]], dtype=mx.int32)
    model._compute_visual_features = fake_compute_visual_features

    with np.testing.assert_raises_regex(ValueError, "image_grid_thw"):
        model.encode_image(pixel_values)

    features = model.encode_image(pixel_values, image_grid_thw=image_grid_thw)

    assert features.tolist() == [[1.0, 1.0, 1.0]]
    assert captured["pixel_values"] is pixel_values
    assert captured["grid_thw"] is image_grid_thw

    video_grid_thw = mx.array([[1, 1, 1]], dtype=mx.int32)
    with np.testing.assert_raises_regex(ValueError, "video_grid_thw"):
        model.encode_video(pixel_values)

    video_features = model.encode_video(pixel_values, video_grid_thw=video_grid_thw)

    assert video_features.tolist() == [[1.0, 1.0, 1.0]]
    assert captured["pixel_values"] is pixel_values
    assert captured["grid_thw"] is video_grid_thw


def test_minimax_m3_get_input_embeddings_uses_configured_token_ids():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=32,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        image_token_index=5,
        video_token_index=6,
        image_token_id=15,
        video_token_id=16,
        projector_hidden_size=3,
        vocab_size=32,
    )
    model = Model(config)
    input_ids = mx.array([[1, 15, 2, 16, 16, 3]], dtype=mx.int32)

    output = model.get_input_embeddings(
        input_ids=input_ids,
        cached_image_features=mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32),
        cached_video_features=mx.array(
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=mx.float32,
        ),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, False, True, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [1.0, 2.0, 3.0]
    assert output.inputs_embeds[0, 3].tolist() == [4.0, 5.0, 6.0]
    assert output.inputs_embeds[0, 4].tolist() == [7.0, 8.0, 9.0]


def test_minimax_m3_sanitize_packs_mxfp8_weights_and_stacks_expert_quant_arrays():
    text_config = _tiny_minimax_text_config(num_hidden_layers=1)
    text_config.moe_layer_freq = [1]
    text_config.num_local_experts = 2
    text_config.n_shared_experts = 0
    config = ModelConfig(
        text_config=text_config,
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
        ),
        projector_hidden_size=3,
        vocab_size=32,
    )
    model = Model(config)

    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": mx.arange(
            16, dtype=mx.uint8
        ).reshape(2, 8),
        "language_model.model.layers.0.self_attn.q_proj.weight_scale_inv": mx.ones(
            (2, 2), dtype=mx.uint8
        ),
    }
    prefix = "language_model.model.layers.0.block_sparse_moe"
    for expert in range(text_config.num_local_experts):
        for old_name in ("w1", "w2", "w3"):
            key = f"{prefix}.experts.{expert}.{old_name}"
            weights[f"{key}.weight"] = mx.full((2, 8), expert + 1, dtype=mx.uint8)
            weights[f"{key}.weight_scale_inv"] = mx.full(
                (2, 2), expert + 2, dtype=mx.uint8
            )
            weights[f"{key}.biases"] = mx.full((2, 2), expert + 4, dtype=mx.float32)

    out = model.sanitize(weights)

    q_weight = out["language_model.model.layers.0.self_attn.q_proj.weight"]
    assert q_weight.dtype == mx.uint32
    assert q_weight.shape == (2, 2)
    assert q_weight[0].tolist() == [
        0 | (1 << 8) | (2 << 16) | (3 << 24),
        4 | (5 << 8) | (6 << 16) | (7 << 24),
    ]
    assert "language_model.model.layers.0.self_attn.q_proj.weight_scale_inv" not in out
    assert "language_model.model.layers.0.self_attn.q_proj.scales" in out

    gate_weight = out[f"{prefix}.switch_mlp.gate_proj.weight"]
    gate_scales = out[f"{prefix}.switch_mlp.gate_proj.scales"]
    gate_biases = out[f"{prefix}.switch_mlp.gate_proj.biases"]
    assert gate_weight.dtype == mx.uint32
    assert gate_weight.shape == (2, 2, 2)
    assert gate_scales.shape == (2, 2, 2)
    assert gate_biases.shape == (2, 2, 2)
    assert gate_scales[0].tolist() == [[2, 2], [2, 2]]
    assert gate_scales[1].tolist() == [[3, 3], [3, 3]]
    assert gate_biases[0].tolist() == [[4.0, 4.0], [4.0, 4.0]]
    assert gate_biases[1].tolist() == [[5.0, 5.0], [5.0, 5.0]]
    assert f"{prefix}.experts.0.w1.weight" not in out
    assert f"{prefix}.experts.0.w1.scales" not in out
    assert f"{prefix}.experts.0.w1.biases" not in out


def test_minimax_m3_get_input_embeddings_uses_cached_video_features():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=16,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        image_token_index=5,
        video_token_index=6,
        projector_hidden_size=3,
        vocab_size=16,
    )
    model = Model(config)

    def fail_compute_visual_features(pixel_values, grid_thw):
        raise AssertionError("video tower should not run for cached video features")

    model._compute_visual_features = fail_compute_visual_features
    output = model.get_input_embeddings(
        input_ids=mx.array([[1, 6, 6, 3]], dtype=mx.int32),
        pixel_values_videos=mx.zeros((2, 3), dtype=mx.float32),
        cached_video_features=mx.array(
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=mx.float32,
        ),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [4.0, 5.0, 6.0]
    assert output.inputs_embeds[0, 2].tolist() == [7.0, 8.0, 9.0]


def test_minimax_m3_get_input_embeddings_uses_cached_image_without_pixels():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=16,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        image_token_index=5,
        video_token_index=6,
        projector_hidden_size=3,
        vocab_size=16,
    )
    model = Model(config)

    def fail_compute_visual_features(pixel_values, grid_thw):
        raise AssertionError("image tower should not run for cached image features")

    model._compute_visual_features = fail_compute_visual_features
    output = model.get_input_embeddings(
        input_ids=mx.array([[1, 5, 3]], dtype=mx.int32),
        cached_image_features=mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [1.0, 2.0, 3.0]


def test_minimax_m3_get_input_embeddings_uses_cached_video_without_pixels():
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=3,
            intermediate_size=4,
            dense_intermediate_size=4,
            shared_intermediate_size=4,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            num_hidden_layers=0,
            vocab_size=16,
            moe_layer_freq=[],
        ),
        vision_config=VisionConfig(
            hidden_size=3,
            intermediate_size=4,
            num_attention_heads=1,
            num_hidden_layers=0,
            img_token_compression_config={
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
        ),
        image_token_index=5,
        video_token_index=6,
        projector_hidden_size=3,
        vocab_size=16,
    )
    model = Model(config)

    def fail_compute_visual_features(pixel_values, grid_thw):
        raise AssertionError("video tower should not run for cached video features")

    model._compute_visual_features = fail_compute_visual_features
    output = model.get_input_embeddings(
        input_ids=mx.array([[1, 6, 6, 3]], dtype=mx.int32),
        cached_video_features=mx.array(
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=mx.float32,
        ),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [4.0, 5.0, 6.0]
    assert output.inputs_embeds[0, 2].tolist() == [7.0, 8.0, 9.0]
