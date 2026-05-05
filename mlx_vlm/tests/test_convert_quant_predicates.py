from types import SimpleNamespace

from mlx_vlm.convert import QUANT_RECIPES, mixed_quant_predicate_builder


class _FakeLinear:
    weight = SimpleNamespace(shape=(128, 128))

    def to_quantized(self):
        return self


class _FakeModel:
    layers = [object() for _ in range(16)]

    def named_modules(self):
        return [("model.layers.0.mlp.down_proj", _FakeLinear())]

    def quant_predicate(self, path, module):
        if "blocked" in path:
            return False
        return True


def test_gemma31_safe_mixed_recipes_are_available():
    assert "mixed_4_6_gemma31_safe" in QUANT_RECIPES
    assert "mixed_4_8_gemma31_safe" in QUANT_RECIPES


def test_gemma31_safe_recipe_keeps_sensitive_paths_higher_precision():
    predicate = mixed_quant_predicate_builder("mixed_4_6_gemma31_safe", _FakeModel())

    assert predicate("model.embed_tokens", _FakeLinear()) == {
        "group_size": 64,
        "bits": 6,
    }
    assert predicate("model.embed_tokens_per_layer", _FakeLinear()) == {
        "group_size": 64,
        "bits": 6,
    }
    assert predicate("model.layers.0.self_attn.v_proj", _FakeLinear()) == {
        "group_size": 64,
        "bits": 6,
    }


def test_gemma31_safe_recipe_uses_four_bits_for_regular_paths():
    predicate = mixed_quant_predicate_builder("mixed_4_6_gemma31_safe", _FakeModel())

    assert predicate("model.layers.2.self_attn.q_proj", _FakeLinear()) == {
        "group_size": 64,
        "bits": 4,
    }


def test_gemma31_safe_recipe_respects_model_quantization_blocks():
    predicate = mixed_quant_predicate_builder("mixed_4_6_gemma31_safe", _FakeModel())

    assert predicate("model.blocked.linear", _FakeLinear()) is False
