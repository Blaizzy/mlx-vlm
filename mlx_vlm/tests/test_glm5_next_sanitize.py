from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.glm5_next.glm5_next import Model


class _IdentitySanitizer:
    def sanitize(self, weights):
        return weights


def _model_for_sanitize():
    sanitizer = _IdentitySanitizer()
    return SimpleNamespace(language_model=sanitizer, vision_tower=sanitizer)


def test_glm5_next_sanitize_remaps_quantized_lm_head_parameters():
    weights = {
        "lm_head.weight": mx.zeros((2, 2)),
        "lm_head.scales": mx.zeros((2, 1)),
        "lm_head.biases": mx.zeros((2, 1)),
    }

    sanitized = Model.sanitize(_model_for_sanitize(), weights)

    assert set(sanitized) == {
        "language_model.lm_head.weight",
        "language_model.lm_head.scales",
        "language_model.lm_head.biases",
    }


def test_glm5_next_sanitize_keeps_remapped_lm_head_stable():
    model = _model_for_sanitize()
    weights = {
        "lm_head.weight": mx.zeros((2, 2)),
        "lm_head.scales": mx.zeros((2, 1)),
        "lm_head.biases": mx.zeros((2, 1)),
    }

    once = Model.sanitize(model, weights)
    twice = Model.sanitize(model, once)

    assert set(twice) == set(once)
    assert all(twice[key] is once[key] for key in once)
