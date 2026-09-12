from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpDecoderLayer


def _tiny_text_config(with_ple=False):
    config = {
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
        "ple_layer_ids": [1] if with_ple else [],
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
    if with_ple:
        config.update(
            {
                "ple_embed_dim": 32,
                "ple_conv_kernel_size": 3,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 17,
                "make_ngram_vocab_size_divisible_by": 4,
                "split_ngram_parts": 4,
                "eos_token_id": 1,
            }
        )
    return TextConfig.from_dict(config)


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


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("prefix_length", [512, 2050, 2051, 2052])
@pytest.mark.parametrize("block_size", [2, 4])
def test_qwen4_qsa_verifier_matches_decode_across_sparse_boundary(
    batch, prefix_length, block_size
):
    from mlx_vlm.models.qwen4_exp.language import (
        BatchQSAKVCache,
        Qwen4ExpAttention,
        Qwen4ExpBatchInvariantForward,
    )

    mx.random.seed(2127)
    config = _tiny_text_config()
    config.hidden_size = 512
    config.num_attention_heads = 4
    config.head_dim = 128
    config.indexer_head_dim = 32
    config.indexer_budget = 2048
    attention = Qwen4ExpAttention(config)
    attention.set_dtype(mx.bfloat16)
    nn.quantize(attention, group_size=32, bits=4)
    verifier = Qwen4ExpBatchInvariantForward()
    caches = [BatchQSAKVCache([0] * batch) for _ in range(2)]
    hidden = mx.random.normal(
        (batch, prefix_length + block_size, config.hidden_size)
    ).astype(mx.bfloat16)
    for cache in caches:
        mx.eval(attention(hidden[:, :prefix_length], cache=cache, mask="causal"))

    # Budget 2048 and compression 4 first select sparse attention at length 2052.
    # Test fully dense blocks, blocks spanning the transition, and sparse blocks.
    proposal = hidden[:, prefix_length:]
    expected = mx.concatenate(
        [
            verifier._qsa_attention(
                attention, proposal[:, index : index + 1], caches[0], None, None
            )
            for index in range(block_size)
        ],
        axis=1,
    )
    actual = verifier._qsa_attention(attention, proposal, caches[1], None, "causal")
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()
    assert caches[0].index_offset == caches[1].index_offset
    assert mx.array_equal(caches[0].index_keys, caches[1].index_keys).item()


def test_qwen4_fused_greedy_mixes_captured_hyper_state_before_lm_head(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as qwen4_language

    config = _tiny_text_config()
    language = LanguageModel(config, _outer_config())
    verifier = qwen4_language._QWEN4_BATCH_INVARIANT_FORWARD
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
