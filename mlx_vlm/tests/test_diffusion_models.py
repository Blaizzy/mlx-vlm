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


class _Detokenizer:
    def __init__(self):
        self.text = ""
        self.offset = 0

    def reset(self):
        self.text = ""
        self.offset = 0

    def add_token(self, token, skip_special_token_ids=None):
        self.text += "decoded"

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class _Processor:
    tokenizer = _Tokenizer()
    detokenizer = _Detokenizer()


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

        self.assertEqual(generate_kwargs["threshold"], 0.7)
        self.assertEqual(generate_kwargs["min_threshold"], 0.7)
        self.assertEqual(generate_kwargs["editing_threshold"], 0.5)
        self.assertEqual(generate_kwargs["max_post_steps"], 16)
        self.assertEqual(generate_kwargs["num_to_transfer"], 1)
        self.assertIsNone(generate_kwargs["max_transfer_per_step"])
        self.assertEqual(generate_kwargs["stability_steps"], 2)
        self.assertEqual(generate_kwargs["block_length"], 32)
        self.assertEqual(generate_kwargs["steps"], 32)
        self.assertEqual(result.text, "decoded")
        self.assertEqual(result.generation_tokens, 3)

        generate_kwargs.clear()
        next(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=input_ids,
                max_tokens=8,
                max_denoising_steps=7,
                block_length=16,
                num_to_transfer=3,
                max_transfer_per_step=2,
                threshold=0.8,
                min_threshold=0.6,
                editing_threshold=0.7,
                max_post_steps=2,
                stability_steps=1,
                temperature=0.0,
            )
        )

        self.assertEqual(generate_kwargs["steps"], 7)
        self.assertEqual(generate_kwargs["block_length"], 16)
        self.assertEqual(generate_kwargs["num_to_transfer"], 3)
        self.assertEqual(generate_kwargs["max_transfer_per_step"], 2)
        self.assertEqual(generate_kwargs["threshold"], 0.8)
        self.assertEqual(generate_kwargs["min_threshold"], 0.6)
        self.assertEqual(generate_kwargs["editing_threshold"], 0.7)
        self.assertEqual(generate_kwargs["max_post_steps"], 2)
        self.assertEqual(generate_kwargs["stability_steps"], 1)

        generate_kwargs.clear()
        next(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=input_ids,
                max_tokens=8,
                num_to_transfer=2,
                temperature=0.0,
            )
        )

        self.assertEqual(generate_kwargs["threshold"], 0.7)
        self.assertEqual(generate_kwargs["min_threshold"], 0.7)
        self.assertEqual(generate_kwargs["editing_threshold"], 0.5)
        self.assertEqual(generate_kwargs["max_post_steps"], 16)
        self.assertEqual(generate_kwargs["num_to_transfer"], 2)

        model = llada2_moe.Model(config)

        self.dtype_consistency_test_runner(
            model.language_model,
            config.model_type,
            config.num_hidden_layers,
        )

    def test_llada_steps_caps_block_denoising(self):
        from mlx_vlm.models import llada2_moe
        from mlx_vlm.models.llada2_moe import language as llada_language

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
        original_call = llada_language.LLaDA2MoeModel.__call__
        calls = {"count": 0}

        def counted_call(self, *args, **kwargs):
            calls["count"] += 1
            return original_call(self, *args, **kwargs)

        llada_language.LLaDA2MoeModel.__call__ = counted_call
        try:
            generated = model.language_model.generate(
                mx.array([[4]], dtype=mx.int32),
                block_length=4,
                steps=1,
                gen_length=8,
                max_post_steps=4,
                mask_id=127,
                eos_id=3,
            )
            mx.eval(generated)
        finally:
            llada_language.LLaDA2MoeModel.__call__ = original_call

        self.assertLessEqual(calls["count"], 6)

    def test_nemotron_labs_diffusion(self):
        from mlx_vlm.models import nemotron_labs_diffusion

        config = nemotron_labs_diffusion.ModelConfig(
            model_type="nemotron_labs_diffusion",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 64,
            },
            eos_token_id=3,
            mask_token_id=127,
        )
        model = nemotron_labs_diffusion.Model(config)
        self.dtype_consistency_test_runner(
            model.language_model,
            config.model_type,
            config.num_hidden_layers,
        )

        generated = model.language_model.generate(
            mx.array([[4]], dtype=mx.int32),
            block_length=4,
            steps=1,
            gen_length=8,
            max_post_steps=4,
            mask_id=127,
            eos_id=999,
        )
        self.assertEqual(generated.shape, (1, 8))

        ar_generated, ar_nfe = model.language_model.ar_generate(
            mx.array([[4]], dtype=mx.int32),
            max_new_tokens=2,
            eos_token_id=3,
        )
        mx.eval(ar_generated)
        self.assertEqual(ar_generated.shape[0], 1)
        self.assertLessEqual(ar_generated.shape[1], 3)
        self.assertGreaterEqual(ar_nfe, 1)

        spec_generated, spec_nfe = model.language_model.linear_spec_generate(
            mx.array([[4]], dtype=mx.int32),
            max_new_tokens=2,
            block_length=2,
            eos_token_id=3,
            mask_token_id=127,
        )
        mx.eval(spec_generated)
        self.assertEqual(spec_generated.shape[0], 1)
        self.assertLessEqual(spec_generated.shape[1], 3)
        self.assertGreaterEqual(spec_nfe, 1)

        def unexpected_diffusion_generate(*args, **kwargs):
            raise AssertionError("Default Nemotron generation should use AR")

        model.language_model.generate = unexpected_diffusion_generate
        default_results = list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4]], dtype=mx.int32),
                max_tokens=1,
                temperature=0.0,
            )
        )
        self.assertEqual(default_results[-1].generation_tokens, 1)

        diffusion_calls = {}

        def diffusion_generate(input_ids, **kwargs):
            diffusion_calls["kwargs"] = kwargs
            kwargs["stats"]["prompt_time"] = 1.0
            return mx.array([[5, 3]], dtype=mx.int32)

        model.language_model.generate = diffusion_generate
        diffusion_results = list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4]], dtype=mx.int32),
                max_tokens=2,
                generation_mode="diffusion",
                temperature=0.0,
            )
        )
        self.assertTrue(diffusion_calls["kwargs"])
        self.assertFalse(diffusion_calls["kwargs"]["linear_speculative"])
        self.assertEqual(diffusion_calls["kwargs"]["steps"], 32)
        self.assertEqual(diffusion_calls["kwargs"]["threshold"], 0.9)
        self.assertEqual(diffusion_results[-1].generation_tokens, 2)

        linear_calls = {}

        def generate(input_ids, **kwargs):
            linear_calls["kwargs"] = kwargs
            kwargs["stats"]["prompt_time"] = 1.0
            return mx.array([[5, 3]], dtype=mx.int32)

        model.language_model.generate = generate
        results = list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4]], dtype=mx.int32),
                max_tokens=2,
                generation_mode="linear_speculative",
                temperature=0.0,
            )
        )
        self.assertTrue(linear_calls["kwargs"])
        self.assertTrue(linear_calls["kwargs"]["linear_speculative"])
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[-1].generation_tokens, 2)
        self.assertEqual(results[-1].finish_reason, "stop")
