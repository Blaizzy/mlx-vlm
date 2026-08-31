import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpDecoderLayer
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter, get_mtp_splitter
from mlx_vlm.speculative.drafters.qwen4_exp_mtp import (
    ModelConfig,
    Qwen4ExpMTPDraftModel,
)
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import split_qwen4_exp_mtp
from mlx_vlm.speculative.mtp import _mtp_next_block_size


def _tiny_text_config():
    return TextConfig.from_dict(
        {
            "model_type": "qwen4_exp_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 16,
            "moe_intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "hc_count": 2,
            "hc_lowrank": 8,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "ple_layer_ids": [],
            "indexer_n_heads": 1,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 16,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [1, 1, 0],
                "rope_theta": 10_000,
                "partial_rotary_factor": 0.25,
            },
            "mtp_num_hidden_layers": 1,
        }
    )


def _outer_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=59,
    )


def test_qwen4_decoder_layers_expose_normalized_layer_types_for_mtp():
    config = _tiny_text_config()

    assert Qwen4ExpDecoderLayer(config, 0).layer_type == "linear_attention"
    assert Qwen4ExpDecoderLayer(config, 1).layer_type == "qwen_sparse_attention"


def test_qwen4_mtp_fusion_matches_released_equations():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    drafter.fc_embedding.weight = mx.eye(config.hidden_size)
    drafter.fc_hidden.weight = mx.eye(config.hidden_size)
    released_embedding_gain = mx.linspace(-0.8, -0.2, config.hidden_size)
    released_hidden_gain = mx.linspace(-0.6, 0.3, config.hc_count * config.hidden_size)
    drafter.pre_fc_norm_embedding.weight = 1 + released_embedding_gain
    drafter.pre_fc_norm_hidden.weight = 1 + released_hidden_gain
    embedding = mx.arange(1, 33, dtype=mx.float32).reshape(1, 1, 32)
    hidden = mx.arange(1, 65, dtype=mx.float32).reshape(1, 1, 64)

    actual = drafter.fuse_inputs(embedding, hidden)
    expected_embedding = embedding * mx.rsqrt(
        mx.mean(embedding * embedding, axis=-1, keepdims=True) + config.rms_norm_eps
    )
    expected_embedding = expected_embedding * (1 + released_embedding_gain)
    expected_hidden = hidden * mx.rsqrt(
        mx.mean(hidden * hidden, axis=-1, keepdims=True) + config.rms_norm_eps
    )
    expected_hidden = expected_hidden * (1 + released_hidden_gain)
    expected_hidden = expected_hidden.reshape(1, 1, config.hc_count, 32)
    expected = (expected_embedding[..., None, :] + expected_hidden).reshape(1, 1, 64)

    assert mx.allclose(actual, expected, atol=2e-5).item()


def test_qwen4_mtp_uses_requested_block_size_as_adaptive_ceiling():
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=_tiny_text_config()))

    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2

    drafter.accept_lens.extend([1] * 8)
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 4

    drafter.accept_lens.extend([0] * 16)
    assert _mtp_next_block_size(drafter, 4, 2, 32) == 2


def test_qwen4_mtp_draft_block_uses_hyper_connection_hidden():
    config = _tiny_text_config()
    drafter = Qwen4ExpMTPDraftModel(ModelConfig(text_config=config))
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            args=config,
            model=SimpleNamespace(embed_tokens=nn.Embedding(64, 32)),
            lm_head=nn.Linear(32, 64, bias=False),
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    tokens = drafter.draft_block(
        7,
        mx.zeros((1, 1, 64)),
        None,
        2,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    mx.eval(tokens)

    assert tokens.shape == (1, 1)
    assert drafter._cache[0].offset == 1


@pytest.mark.parametrize("accepted", [0, 1])
def test_qwen4_target_exposes_pre_mixer_hidden_and_replays_rejection_exactly(
    accepted,
):
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5, 6]], dtype=mx.int32)

    speculative_cache = language.make_cache()
    prefill = language(prompt, cache=speculative_cache, return_hidden=True)
    hidden, _, rollback = language.speculative_verify_hidden(verify, speculative_cache)
    language.rollback_speculative_cache(
        speculative_cache, rollback, accepted=accepted, block_size=3
    )

    reference_cache = language.make_cache()
    language(prompt, cache=reference_cache)
    for index in range(accepted + 1):
        language(verify[:, index : index + 1], cache=reference_cache)
    probe = mx.array([[7]], dtype=mx.int32)
    speculative_logits = language(probe, cache=speculative_cache).logits
    reference_logits = language(probe, cache=reference_cache).logits
    mx.eval(prefill.hidden_states, hidden, speculative_logits, reference_logits)

    assert prefill.hidden_states[-1].shape == (1, 3, 64)
    assert hidden.shape == (1, 3, 64)
    assert mx.array_equal(speculative_logits, reference_logits).item()


def test_qwen4_speculative_verifier_matches_tokenwise_hidden_and_logits():
    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    verify = mx.array([[4, 5, 6]], dtype=mx.int32)

    batched_cache = language.make_cache()
    language(prompt, cache=batched_cache)
    batched_hidden, _, _, batched_logits = language.speculative_verify_logits(
        verify, batched_cache, lambda logits: logits
    )

    tokenwise_cache = language.make_cache()
    language(prompt, cache=tokenwise_cache)
    tokenwise_hidden = []
    tokenwise_logits = []
    for index in range(verify.shape[1]):
        output = language(
            verify[:, index : index + 1],
            cache=tokenwise_cache,
            return_hidden=True,
        )
        tokenwise_hidden.append(output.hidden_states[-1])
        tokenwise_logits.append(output.logits)
    tokenwise_hidden = mx.concatenate(tokenwise_hidden, axis=1)
    tokenwise_logits = mx.concatenate(tokenwise_logits, axis=1)
    mx.eval(batched_hidden, batched_logits, tokenwise_hidden, tokenwise_logits)

    assert mx.allclose(batched_hidden, tokenwise_hidden, rtol=0, atol=1e-6).item()
    assert mx.allclose(batched_logits, tokenwise_logits, rtol=0, atol=1e-6).item()
    assert mx.array_equal(
        mx.argmax(batched_logits, axis=-1),
        mx.argmax(tokenwise_logits, axis=-1),
    ).item()


def test_qwen4_fused_greedy_mixes_captured_hyper_state_before_lm_head(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as qwen4_language

    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    verifier = qwen4_language._QWEN4_EXACT_SPECULATIVE_VERIFIER
    monkeypatch.setattr(verifier, "can_quantized_head", lambda linear: True)
    monkeypatch.setattr(
        verifier,
        "quantized_argmax",
        lambda linear, hidden, token_mask=None: mx.argmax(linear(hidden), axis=-1),
    )
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = mx.argmax(language(inputs, cache=language.make_cache()).logits, axis=-1)
    language._position_ids = None
    language._rope_deltas = None
    actual = language.fused_greedy_decode(inputs, cache=language.make_cache())
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_mtp_splitter_maps_fused_experts_and_quantizes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": text_config})
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((64,)),
            "mtp.fc_hidden.weight": mx.ones((32, 32)),
            "mtp.layers.0.mlp.experts.gate_up_proj": mx.ones((4, 32, 32)),
            "mtp.layers.0.mlp.experts.down_proj": mx.ones((4, 32, 16)),
            "mtp.layers.0.mlp.gate.weight": mx.ones((4, 32)),
        },
    )

    splitter = detect_mtp_splitter(source)
    assert splitter is not None
    assert splitter.output_model_type == "qwen4_exp_mtp"
    assert get_mtp_splitter("qwen4_exp").output_model_type == "qwen4_exp_mtp"

    split_qwen4_exp_mtp(str(source), str(output), q_bits=3, q_group_size=32)
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())

    assert "layers.0.mlp.switch_mlp.gate_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.up_proj.weight" in weights
    assert "layers.0.mlp.switch_mlp.down_proj.weight" in weights
    assert "fc_hidden.scales" in weights
    assert "layers.0.mlp.gate.scales" not in weights
    assert config["model_type"] == "qwen4_exp_mtp"
    assert config["block_size"] == 2
    assert config["quantization"] == {
        "group_size": 32,
        "bits": 3,
        "mode": "affine",
    }


def test_qwen4_mtp_splitter_converts_official_fp8_experts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_text_config().to_dict()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": text_config,
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    weights = {"mtp.pre_fc_norm_hidden.weight": mx.zeros((64,))}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
            weights[key] = mx.to_fp8(mx.ones((128, 128)) * (expert + 1))
            weights[f"{key}_scale_inv"] = mx.ones((1, 1))
    mx.save_safetensors(str(source / "model.safetensors"), weights)

    split_qwen4_exp_mtp(str(source), str(output))

    split_weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    gate = split_weights["layers.0.mlp.switch_mlp.gate_proj.weight"]
    up = split_weights["layers.0.mlp.switch_mlp.up_proj.weight"]
    down = split_weights["layers.0.mlp.switch_mlp.down_proj.weight"]
    mx.eval(gate, up, down)
    assert gate.shape == (2, 128, 128)
    assert up.shape == (2, 128, 128)
    assert down.shape == (2, 128, 128)
    assert not any(key.endswith("weight_scale_inv") for key in split_weights)
    assert "quantization" not in config
