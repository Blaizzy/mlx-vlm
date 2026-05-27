import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.generate.dispatch import stream_generate


class _StoppingCriteria:
    def __call__(self, token):
        return token == 3


class _Tokenizer:
    stopping_criteria = _StoppingCriteria()

    def decode(self, tokens, skip_special_tokens=False):
        return "decoded"


class _Processor:
    tokenizer = _Tokenizer()


class _LanguageModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        kwargs["stats"]["prompt_time"] = 1.0
        return mx.array([[1, 2, 3]], dtype=mx.int32)


class _LLaDAModel:
    config = SimpleNamespace(model_type="llada2_moe", image_token_index=None)

    def __init__(self):
        self.language_model = _LanguageModel()


class TestDiffusionModels(unittest.TestCase):
    def test_llada_dispatch_uses_quality_defaults(self):
        model = _LLaDAModel()
        input_ids = mx.array([[4, 5]], dtype=mx.int32)

        result = next(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=input_ids,
                max_tokens=8,
                temperature=0.0,
            )
        )

        kwargs = model.language_model.kwargs
        self.assertEqual(kwargs["threshold"], 0.95)
        self.assertEqual(kwargs["editing_threshold"], 0.9)
        self.assertEqual(kwargs["max_post_steps"], 4)
        self.assertEqual(result.text, "decoded")
        self.assertEqual(result.generation_tokens, 3)
