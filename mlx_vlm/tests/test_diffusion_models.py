import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_vlm.generate.dispatch import stream_generate
from mlx_vlm.models.cache import StaticPrefixKVCache


class _StoppingCriteria:
    def __call__(self, token):
        return token == 3


class _Tokenizer:
    stopping_criteria = _StoppingCriteria()

    def decode(self, tokens, skip_special_tokens=False):
        return "decoded"


class _Processor:
    tokenizer = _Tokenizer()


class TestDiffusionModels(unittest.TestCase):
    def dtype_consistency_test_runner(self, language_model, model_type, num_layers):
        self.assertEqual(language_model.model_type, model_type)
        self.assertEqual(len(language_model.layers), num_layers)
        inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

        for dtype in [mx.float32, mx.float16]:
            with self.subTest(dtype=dtype):
                language_model.update(
                    tree_map(lambda p: p.astype(dtype), language_model.parameters())
                )

                outputs = language_model(inputs)
                self.assertEqual(outputs.logits.dtype, dtype)

                prefix_cache = [
                    StaticPrefixKVCache(max_size=8) for _ in language_model.layers
                ]
                mx.eval(language_model.model(inputs[:, :2], cache=prefix_cache))
                block_cache = [StaticPrefixKVCache.from_prefix(c) for c in prefix_cache]
                cached_outputs = language_model(inputs[:, 2:], cache=block_cache)
                self.assertEqual(cached_outputs.logits.dtype, dtype)

    def test_llada(self):
        from mlx_vlm.models import llada2_moe

        config = llada2_moe.ModelConfig(
            model_type="llada2_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rotary_dim=8,
            num_experts=None,
            max_position_embeddings=128,
            pad_token_id=3,
            eos_token_id=3,
            mask_token_id=127,
        )
        model = llada2_moe.Model(config)
        generate_kwargs = {}

        def generate(input_ids, **kwargs):
            generate_kwargs.update(kwargs)
            kwargs["stats"]["prompt_time"] = 1.0
            return mx.array([[1, 2, 3]], dtype=mx.int32)

        model.language_model.generate = generate
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

        self.assertEqual(generate_kwargs["threshold"], 0.95)
        self.assertEqual(generate_kwargs["editing_threshold"], 0.9)
        self.assertEqual(generate_kwargs["max_post_steps"], 4)
        self.assertEqual(result.text, "decoded")
        self.assertEqual(result.generation_tokens, 3)

        model = llada2_moe.Model(config)

        self.dtype_consistency_test_runner(
            model.language_model,
            config.model_type,
            config.num_hidden_layers,
        )
