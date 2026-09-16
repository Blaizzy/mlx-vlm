import mlx.core as mx
import numpy as np
import pytest

import mlx_vlm.models.minimax_m3_vl.language as minimax_language
from mlx_vlm.models.minimax_m3_vl.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.minimax_m3_vl.language import (
    LanguageModel,
    MiniMaxAttention,
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
    MiniMaxSparseMoeBlock,
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


def _minimax_text_config(**overrides):
    """Build a fresh config; callers specify only scenario-specific differences."""
    fields = dict(
        hidden_size=4,
        intermediate_size=4,
        dense_intermediate_size=4,
        shared_intermediate_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        num_hidden_layers=1,
        vocab_size=8,
    )
    fields.update(overrides)
    return TextConfig(**fields)


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
        num_hidden_layers=1, sparse_attention_config={"use_sparse_attention": False}
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


def test_minimax_m3_language_model_initial_batch_prefill_uses_rope_index():
    config = _tiny_minimax_text_config(num_hidden_layers=1)
    language_model = LanguageModel(config)
    captured = {}

    class ModelStub:
        def __call__(self, inputs, **kwargs):
            captured["mask"] = kwargs["mask"]
            captured["position_ids"] = kwargs["position_ids"]
            return mx.zeros(
                (inputs.shape[0], inputs.shape[1], config.hidden_size), dtype=mx.float32
            )

    object.__setattr__(language_model, "model", ModelStub())
    cache = [MiniMaxM3BatchKVCache([2, 0])]
    input_ids = mx.array([[0, 0, 5, 6], [7, 8, 9, 10]], dtype=mx.int32)

    language_model(input_ids, cache=cache, skip_logits=True)
    mx.eval(captured["mask"], captured["position_ids"])

    assert captured["mask"].tolist() == [
        [False, False, True, True],
        [True, True, True, True],
    ]
    assert captured["position_ids"].tolist() == [[-2, -1, 0, 1], [0, 1, 2, 3]]
    assert language_model._rope_deltas.tolist() == [[0], [0]]


def test_minimax_m3_language_model_decode_uses_batch_cache_offsets():
    config = _tiny_minimax_text_config(num_hidden_layers=1)
    language_model = LanguageModel(config)
    language_model._rope_deltas = mx.zeros((2, 1), dtype=mx.int32)
    captured = {}

    class ModelStub:
        def __call__(self, inputs, **kwargs):
            captured["position_ids"] = kwargs["position_ids"]
            return mx.zeros(
                (inputs.shape[0], inputs.shape[1], config.hidden_size), dtype=mx.float32
            )

    object.__setattr__(language_model, "model", ModelStub())
    cache = MiniMaxM3BatchKVCache([2, 0])
    cache.kv_cache.offset = mx.array([2, 4], dtype=mx.int32)
    cache.kv_cache._idx = 4

    language_model(
        mx.array([[11, 12], [13, 14]], dtype=mx.int32), cache=[cache], skip_logits=True
    )
    mx.eval(captured["position_ids"])

    assert captured["position_ids"].tolist() == [[2, 3], [4, 5]]


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


def test_minimax_m3_config_coerces_direct_dict_subconfigs():
    config = ModelConfig(
        text_config={"num_hidden_layers": 2, "mlp_layer_types": ["dense", "sparse"]},
        vision_config={
            "img_token_compression_config": {
                "spatial_merge_size": 3,
                "temporal_patch_size": 1,
            }
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
    ] == {"bits": 8, "group_size": 64}
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


def test_minimax_m3_prompt_utils_formats_text_only_messages():
    result = apply_chat_template(
        None,
        {"model_type": "minimax_m3"},
        "Write a short answer.",
        return_messages=True,
        num_images=1,
    )

    assert result == [{"role": "user", "content": "Write a short answer."}]


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
            encoded = [
                ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded
            ]
        attention_mask = [
            [0 if token == self.pad_token_id else 1 for token in ids] for ids in encoded
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
    expected_text = (
        f"]<]0.0 seconds[>[{frame_chunk}]<]1.0 seconds[>[{frame_chunk}Describe."
    )
    assert processor.tokenizer.seen_text == [expected_text]
    assert outputs["pixel_values_videos"].shape == (8, 24)
    assert outputs["video_grid_thw"].tolist() == [[2, 2, 2]]
    assert outputs["input_ids"][0].count(processor.video_token_id) == 2
    for index, token_id in enumerate(outputs["input_ids"][0]):
        if token_id == processor.video_token_id:
            assert outputs["mm_token_type_ids"][0][index] == 2


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


def test_minimax_m3_batch_rollback_raises_on_ragged_accepts():
    # Ragged accepts on a rectangular batch cache would zero the shorter row's
    # KV and indexer tails, leaving phantom keys attended (issue #1962). The
    # rollback must fail loud so callers clamp to uniform acceptance instead.
    lm = LanguageModel(_tiny_minimax_text_config(num_hidden_layers=1))
    cache = MiniMaxM3BatchKVCache([0, 0])
    keys = mx.ones((2, 1, 5, 4), dtype=mx.float32)
    values = mx.ones((2, 1, 5, 4), dtype=mx.float32)
    index_keys = mx.ones((2, 1, 5, 4), dtype=mx.float32)

    cache.update_and_fetch(keys, values)
    cache.update_index_and_fetch(index_keys)
    with pytest.raises(RuntimeError, match="uniform"):
        lm.rollback_speculative_cache(
            [cache], None, accepted=mx.array([2, 0], dtype=mx.int32), block_size=4
        )


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


def test_minimax_m3_batch_cache_merge_preserves_index_cache_rows():
    warm = MiniMaxM3KVCache()
    warm.update_and_fetch(
        mx.ones((1, 1, 3, 4), dtype=mx.float32), mx.ones((1, 1, 3, 5), dtype=mx.float32)
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


def test_minimax_m3_uniform_batch_cache_skips_decode_mask():
    cache = MiniMaxM3BatchKVCache([0, 0])
    generic_cache = minimax_language.BatchKVCache([0, 0])

    assert cache.make_mask(1) is None
    assert cache.make_mask(2) is not None
    assert not minimax_language._can_skip_uniform_batch_decode_mask(generic_cache, 1)
    assert minimax_language._can_skip_uniform_batch_decode_mask(
        [generic_cache, cache], 1
    )
    assert not minimax_language._can_skip_uniform_batch_decode_mask(generic_cache, 2)
    assert MiniMaxM3BatchKVCache([1, 0]).make_mask(1) is not None
    assert not minimax_language._can_skip_uniform_batch_decode_mask(
        [minimax_language.BatchKVCache([1, 0]), MiniMaxM3BatchKVCache([1, 0])], 1
    )


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

    config = _minimax_text_config(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=32,
        shared_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rotary_dim=4,
        vocab_size=32,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 1,
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

    config = _minimax_text_config(
        hidden_size=12,
        intermediate_size=8,
        dense_intermediate_size=24,
        shared_intermediate_size=8,
        num_attention_heads=3,
        head_dim=4,
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


def test_minimax_m3_sparse_selection_merges_2d_causal_mask_on_query_axis():
    config = _minimax_text_config(
        rotary_dim=1,
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
    mask = mx.array([[True, False, False], [True, True, False], [True, True, True]])

    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, 0, mask)

    assert sparse_mask.shape == (1, 1, 3, 3)
    assert sparse_mask.tolist() == [
        [[[True, False, False], [True, True, False], [True, True, True]]]
    ]


def test_minimax_m3_dense_attention_combines_integer_padding_with_causal_mask(
    monkeypatch,
):
    config = _minimax_text_config(
        head_dim=4,
        rotary_dim=4,
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
        mx.ones((1, 3, 4), dtype=mx.float32), mask=mx.array([[1, 1, 0]], dtype=mx.int32)
    )
    mx.eval(output)

    assert captured["mask"].shape == (1, 1, 3, 3)
    assert captured["mask"].tolist() == [
        [[[True, False, False], [True, True, False], [True, True, False]]]
    ]


def test_minimax_m3_direct_sparse_decode_matches_sparse_masked_attention():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return

    config = _minimax_text_config(
        hidden_size=128,
        intermediate_size=64,
        dense_intermediate_size=64,
        shared_intermediate_size=64,
        num_attention_heads=4,
        head_dim=32,
        rotary_dim=32,
        vocab_size=16,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 32,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16,
            "sparse_init_block": 1,
            "sparse_local_block": 2,
        },
    )
    attention = MiniMaxAttention(config, 0)
    rng = np.random.default_rng(13)
    queries = mx.array(
        rng.normal(size=(1, 4, 1, 32)).astype(np.float16), dtype=mx.float16
    )
    keys = mx.array(
        rng.normal(size=(1, 1, 131072, 32)).astype(np.float16), dtype=mx.float16
    )
    values = mx.array(
        rng.normal(size=(1, 1, 131072, 32)).astype(np.float16), dtype=mx.float16
    )
    q_positions = mx.array([[131071]], dtype=mx.int32)
    block_indices = mx.arange(16, dtype=mx.int32)[None, None, :]
    mask = attention.indexer.build_block_mask(
        block_indices, None, keys.shape[2], queries.dtype, q_positions
    )

    dense = minimax_language.scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=attention.scale, mask=mask
    )
    sparse = attention._sparse_decode_attention(
        queries, keys, values, block_indices, "causal", q_positions
    )
    mx.eval(dense, sparse)

    assert sparse is not None
    np.testing.assert_allclose(np.array(sparse), np.array(dense), rtol=2e-2, atol=2e-2)


def test_minimax_m3_short_decode_keeps_full_sparse_kv(monkeypatch):
    config = _minimax_text_config(
        num_attention_heads=2,
        head_dim=2,
        rotary_dim=2,
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

    mx.eval(attention(mx.ones((1, 5, 4), dtype=mx.float32), cache=cache))
    mx.eval(attention(mx.ones((1, 1, 4), dtype=mx.float32), cache=cache))

    assert key_lengths[-2:] == [5, 6]


def test_minimax_m3_batched_decode_compacts_sparse_kv(monkeypatch):
    config = _minimax_text_config(
        num_attention_heads=2,
        head_dim=2,
        rotary_dim=2,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 2,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 1,
            "sparse_topk_blocks": 1,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
        },
    )
    attention = MiniMaxAttention(config, 0)
    cache = MiniMaxM3BatchKVCache([2, 0])
    key_lengths = []

    def fake_sdpa(queries, keys, values, cache, scale, mask):
        del values, cache, scale, mask
        key_lengths.append(keys.shape[2])
        return mx.zeros_like(queries)

    monkeypatch.setattr(minimax_language, "scaled_dot_product_attention", fake_sdpa)

    mx.eval(
        attention(
            mx.ones((2, 20, 4), dtype=mx.float32),
            cache=cache,
            position_ids=mx.array(
                [
                    [
                        -2,
                        -1,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                    ],
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                    ],
                ],
                dtype=mx.int32,
            ),
        )
    )
    mx.eval(
        attention(
            mx.ones((2, 1, 4), dtype=mx.float32),
            mask=cache.make_mask(1),
            cache=cache,
            position_ids=mx.array([[18], [20]], dtype=mx.int32),
        )
    )

    assert key_lengths[-2:] == [20, 1]


def test_minimax_m3_compiled_sparse_indexer_matches_generic_selection():
    config = _minimax_text_config(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        rotary_dim=2,
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
    idx_queries = mx.array([[[[1.0, 0.25]], [[0.25, 1.0]]]], dtype=mx.float32)
    idx_keys = mx.array(
        [[[[10.0, 0.0], [9.0, 0.0], [0.0, 10.0], [0.0, 9.0], [3.0, 3.0], [2.0, 2.0]]]],
        dtype=mx.float32,
    )

    q_start = idx_keys.shape[2] - 1
    qpos = mx.array([[q_start]], dtype=mx.int32)
    kpos = mx.arange(idx_keys.shape[2])
    causal_mask = (kpos[None, :] <= qpos[:, :, None])[None]

    compiled = attention.indexer.select_blocks(idx_queries, idx_keys, q_start=q_start)
    generic = attention.indexer.select_blocks(
        idx_queries, idx_keys, q_start=q_start, mask=causal_mask
    )
    mx.eval(compiled, generic)

    assert compiled.tolist() == generic.tolist()


def test_minimax_m3_sparse_selection_ignores_nan_block_scores():
    config = _minimax_text_config(
        rotary_dim=1,
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


def test_minimax_m3_sparse_lse_selection_uses_attention_head_scale():
    config = _minimax_text_config(
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        head_dim=4,
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


def test_minimax_m3_compiled_sparse_selection_handles_batched_index_heads():
    config = _minimax_text_config(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=16,
        shared_intermediate_size=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=2,
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
        np.array(out.logits), np.array(expected.logits), rtol=1e-6, atol=1e-6
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

    assert selected.tolist() == [[3.0, 4.0, 30.0, 40.0], [5.0, 6.0, 50.0, 60.0]]


def test_minimax_m3_get_input_embeddings_handles_images_and_videos():
    config = ModelConfig(
        text_config=_minimax_text_config(
            hidden_size=3,
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
        text_config=_minimax_text_config(
            hidden_size=3,
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
        text_config=_minimax_text_config(
            hidden_size=3,
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
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float32
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


def test_minimax_m3_get_input_embeddings_uses_cached_image_without_pixels():
    config = ModelConfig(
        text_config=_minimax_text_config(
            hidden_size=3,
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
        text_config=_minimax_text_config(
            hidden_size=3,
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
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float32
        ),
    )

    assert output.visual_pos_masks.tolist() == [[False, True, True, False]]
    assert output.inputs_embeds[0, 1].tolist() == [4.0, 5.0, 6.0]
    assert output.inputs_embeds[0, 2].tolist() == [7.0, 8.0, 9.0]


def test_sparse_index_helpers_are_not_shape_compiled():
    """The sparse index helpers must stay uncompiled.

    Both take `idx_keys` straight from `update_index_and_fetch`, whose sequence
    dimension is the running context length. A non-shapeless `mx.compile` keys
    on input shapes, so it re-traces every decode step: the compiled artifact is
    never reused and each trace leaves permanent state in MLX's compiler cache
    that `mx.clear_cache()` does not release, eventually tripping
    `[metal::malloc] Resource limit (499000) exceeded` on a long-running server.

    If either of these ever needs compiling again, it has to be `shapeless=True`
    with the shape-dependent work (`idx_keys.shape[2]`, `mx.arange(total_len)`)
    lifted out first.
    """

    def is_compiled(fn):
        return type(fn).__name__ == "gc_func"

    assert not is_compiled(minimax_language._build_sparse_causal_mask_compiled)
    assert not is_compiled(minimax_language._select_sparse_block_indices_compiled)

    # Guard the check itself: a compiled function must be detected as one.
    assert is_compiled(mx.compile(lambda x: x + 1))
