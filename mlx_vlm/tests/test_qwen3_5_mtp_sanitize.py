from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model as MoEModel


def _model_for_sanitize(model_class):
    model = model_class.__new__(model_class)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=0)
    )
    return model


def _weights_with_draft_shard():
    return {
        "language_model.model.layers.0.input_layernorm.weight": mx.array([2.0]),
        "language_model.mtp.layers.0.input_layernorm.weight": mx.array([1.0]),
    }


def test_moe_qwen_mtp_shard_does_not_shift_base_norm_weights():
    weights = _model_for_sanitize(MoEModel).sanitize(_weights_with_draft_shard())

    assert weights["language_model.model.layers.0.input_layernorm.weight"].tolist() == [
        2.0
    ]
    assert not any("mtp." in key for key in weights)
