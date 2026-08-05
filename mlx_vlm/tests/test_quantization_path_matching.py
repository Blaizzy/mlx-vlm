from copy import deepcopy

from mlx_vlm.quantization.one_bit import _quantization_for_path
from mlx_vlm.utils import get_class_predicate


class _QuantizableModule:
    class _Weight:
        size = 64

    weight = _Weight()

    def to_quantized(self):
        pass


def _predicate(quantization):
    return get_class_predicate(quantization_config=quantization)


def test_exact_path_uses_matching_override():
    path = "model.layers.0.attn.wkv"
    exact = {"group_size": 16, "bits": 2, "mode": "affine"}
    quantization = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        path: exact,
    }

    assert _quantization_for_path(quantization, path) == exact
    resolved = _predicate(quantization)(path, _QuantizableModule())
    assert resolved == exact
    assert resolved is not exact
    resolved["bits"] = 8
    assert quantization[path] == exact


def test_wrapped_model_layers_use_unwrapped_override():
    path = "language_model.model.layers.1.attn.wkv"
    override = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "model.layers.1.attn.wkv": override,
    }

    assert _quantization_for_path(quantization, path) == override
    assert _predicate(quantization)(path, _QuantizableModule()) == override


def test_wrapped_lm_head_uses_unwrapped_override():
    path = "language_model.lm_head"
    override = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {
        "group_size": 64,
        "bits": 2,
        "mode": "affine",
        "lm_head": override,
    }

    assert _quantization_for_path(quantization, path) == override
    assert _predicate(quantization)(path, _QuantizableModule()) == override


def test_missing_override_uses_global_fallback():
    quantization = {"group_size": 64, "bits": 2, "mode": "affine"}
    path = "language_model.model.layers.7.attn.wkv"

    assert _quantization_for_path(quantization, path) == quantization
    assert _predicate(quantization)(path, _QuantizableModule()) is True


def test_exact_prefixed_override_wins_over_unwrapped_override():
    path = "language_model.model.layers.2.attn.wkv"
    exact = {"group_size": 16, "bits": 2, "mode": "affine"}
    fallback = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        path: exact,
        "model.layers.2.attn.wkv": fallback,
    }

    assert _quantization_for_path(quantization, path) == exact
    assert _predicate(quantization)(path, _QuantizableModule()) == exact


def test_mixed_attention_shared_and_routed_expert_modes():
    attention_path = "language_model.model.layers.3.attn.wkv"
    shared_path = "language_model.model.layers.3.ffn.shared_experts.up_proj"
    routed_path = "language_model.model.layers.3.ffn.switch_mlp.0.up_proj"
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    affine = {"group_size": 64, "bits": 2, "mode": "affine"}
    quantization = {
        **affine,
        "model.layers.3.attn.wkv": mxfp8,
        "model.layers.3.ffn.shared_experts.up_proj": mxfp8,
        "model.layers.3.ffn.switch_mlp.0.up_proj": affine,
    }
    original = deepcopy(quantization)

    for path, expected in (
        (attention_path, mxfp8),
        (shared_path, mxfp8),
        (routed_path, affine),
    ):
        assert _quantization_for_path(quantization, path) == expected
        assert _predicate(quantization)(path, _QuantizableModule()) == expected

    assert quantization == original
