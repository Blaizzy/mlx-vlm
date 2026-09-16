from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.openai_privacy_filter import Model, ModelConfig
from mlx_vlm.privacy_filter import PrivacyFilter

LABELS = {0: "O", 1: "B-x", 2: "I-x", 3: "E-x", 4: "S-x"}


def _tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "sliding_window": 2,
        "max_position_embeddings": 32,
        "default_n_ctx": 32,
        "pad_token_id": 31,
        "eos_token_id": 31,
        "num_labels": 5,
        "id2label": LABELS,
        "attention_chunk_size": 3,
        "moe_chunk_size": 3,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_checkpoint_expert_layout_sanitization():
    gate_up = mx.arange(2 * 3 * 8).reshape(2, 3, 8)
    down = mx.arange(2 * 4 * 3).reshape(2, 4, 3)
    weights = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.gate_up_proj_bias": mx.zeros((2, 8)),
        "model.layers.0.mlp.experts.down_proj": down,
        "model.layers.0.mlp.experts.down_proj_bias": mx.zeros((2, 3)),
        "score.weight": mx.zeros((5, 3)),
    }

    sanitized = Model.sanitize(None, weights)

    gate_key = "model.layers.0.mlp.experts.gate_proj.weight"
    up_key = "model.layers.0.mlp.experts.up_proj.weight"
    down_key = "model.layers.0.mlp.experts.down_proj.weight"
    transposed = gate_up.swapaxes(-1, -2)
    assert sanitized[gate_key].shape == (2, 4, 3)
    assert sanitized[up_key].shape == (2, 4, 3)
    assert sanitized[down_key].shape == (2, 3, 4)
    assert mx.array_equal(sanitized[gate_key], transposed[:, :4]).item()
    assert mx.array_equal(sanitized[up_key], transposed[:, 4:]).item()
    assert mx.array_equal(sanitized[down_key], down.swapaxes(-1, -2)).item()
    assert "model.layers.0.mlp.experts.gate_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.up_proj.bias" in sanitized
    assert "model.layers.0.mlp.experts.down_proj.bias" in sanitized
    assert "score.weight" in sanitized


@pytest.mark.parametrize(
    ("group_size", "bits", "mode"),
    [
        (64, 4, "affine"),
        (64, 5, "affine"),
        (64, 6, "affine"),
        (64, 8, "affine"),
        (32, 4, "mxfp4"),
        (16, 4, "nvfp4"),
        (32, 8, "mxfp8"),
    ],
)
def test_published_quantization_modes_run_end_to_end(group_size, bits, mode):
    model = Model(
        _tiny_config(
            hidden_size=64, intermediate_size=64, head_dim=16, num_attention_heads=4
        )
    )
    nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    logits = model(input_ids, attention_mask=mx.ones_like(input_ids)).logits
    mx.eval(logits)

    assert logits.shape == (1, 5, 5)
    assert mx.all(mx.isfinite(logits)).item()


def test_tiny_forward_supports_attention_and_moe_chunk_boundaries():
    model = Model(_tiny_config())
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    logits = model(input_ids, attention_mask=mx.ones_like(input_ids)).logits
    mx.eval(logits)

    assert logits.shape == (1, 5, 5)
    assert mx.all(mx.isfinite(logits)).item()


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "Alice emailed bob@example.com"
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 5), (5, 13), (13, 29)]}


class _FakeModel:
    def __init__(self):
        labels = {
            0: "O",
            1: "B-private_person",
            2: "I-private_person",
            3: "E-private_person",
            4: "S-private_person",
            5: "B-private_email",
            6: "I-private_email",
            7: "E-private_email",
            8: "S-private_email",
        }
        self.config = SimpleNamespace(
            id2label=labels, num_labels=len(labels), default_n_ctx=16
        )

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        logits = mx.full((1, input_ids.shape[1], 9), -10.0)
        logits[0, 0, 4] = 10.0
        logits[0, 1, 0] = 10.0
        logits[0, 2, 8] = 10.0
        return SimpleNamespace(logits=logits)


def test_high_level_api_returns_offsets_and_redacted_text():
    detector = PrivacyFilter(_FakeModel(), _FakeTokenizer())

    result = detector("Alice emailed bob@example.com")

    assert [(span.label, span.start, span.end) for span in result.spans] == [
        ("private_person", 0, 5),
        ("private_email", 14, 29),
    ]
    assert result.redacted_text == ("<PRIVATE_PERSON> emailed <PRIVATE_EMAIL>")
    assert result.to_dict()["spans"][0]["text"] == "Alice"
