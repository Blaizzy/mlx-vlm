import importlib
import math
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

        # Without explicit overrides the model generate()'s own reference
        # defaults apply; the dispatcher must not force shared tuned values.
        for key in (
            "block_length",
            "steps",
            "threshold",
            "min_threshold",
            "editing_threshold",
            "max_post_steps",
            "num_to_transfer",
            "max_transfer_per_step",
            "stability_steps",
        ):
            self.assertNotIn(key, generate_kwargs)
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

        self.assertNotIn("threshold", generate_kwargs)
        self.assertNotIn("editing_threshold", generate_kwargs)
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

    def test_llada_stream_generate_ignores_extra_cli_kwargs(self):
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

        results = list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4]], dtype=mx.int32),
                max_tokens=2,
                steps=1,
                fps=2.0,
                temperature=0.0,
            )
        )

        self.assertEqual(results[-1].generation_tokens, 2)

        with self.assertRaisesRegex(ValueError, "does not support linear_speculative"):
            list(
                stream_generate(
                    model,
                    _Processor(),
                    prompt="ignored",
                    input_ids=mx.array([[4]], dtype=mx.int32),
                    max_tokens=2,
                    generation_mode="diffusion",
                    linear_speculative=True,
                    temperature=0.0,
                )
            )

    def test_nemotron_labs_diffusion(self):
        from mlx_vlm.models import nemotron_labs_diffusion
        from mlx_vlm.models.nemotron_labs_diffusion.language import (
            _chunked_greedy_score_weight,
        )

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

        model.language_model.update(
            tree_map(lambda p: p.astype(mx.bfloat16), model.language_model.parameters())
        )
        bf16_outputs = model.language_model(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            attention_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
        )
        self.assertEqual(bf16_outputs.logits.dtype, mx.bfloat16)
        bf16_filtered = model.language_model._top_k_logits(bf16_outputs.logits, 2)
        self.assertEqual(bf16_filtered.dtype, mx.bfloat16)
        _, bf16_probs = model.language_model._sample_with_temperature_topk_topp(
            bf16_outputs.logits
        )
        self.assertEqual(bf16_probs.dtype, mx.bfloat16)

        score_hidden = mx.random.normal((1, 3, 16)).astype(mx.float32)
        score_weight = mx.random.normal((4096, 16)).astype(mx.float32)
        score_tokens, score_probs = _chunked_greedy_score_weight(
            score_weight, score_hidden, chunks=16, return_prob=True
        )
        score_logits = score_hidden @ score_weight.T
        ref_tokens = mx.argmax(score_logits, axis=-1).astype(mx.int32)
        ref_logits = mx.take_along_axis(score_logits, ref_tokens[..., None], axis=-1)[
            ..., 0
        ]
        ref_probs = mx.exp(ref_logits - mx.logsumexp(score_logits, axis=-1))
        self.assertEqual(score_tokens.tolist(), ref_tokens.tolist())
        self.assertTrue(bool(mx.allclose(score_probs, ref_probs).item()))

        diffusion_stats = {}
        generated = model.language_model.generate(
            mx.array([[4]], dtype=mx.int32),
            block_length=4,
            steps=1,
            gen_length=8,
            max_post_steps=4,
            mask_id=127,
            eos_id=999,
            stats=diffusion_stats,
        )
        self.assertEqual(generated.shape, (1, 8))
        self.assertEqual(diffusion_stats["diffusion_sampler"], "native")
        self.assertTrue(math.isnan(diffusion_stats["diffusion_min_threshold"]))
        self.assertEqual(diffusion_stats["diffusion_transformers_parity"], 1.0)
        self.assertGreaterEqual(diffusion_stats["diffusion_denoise_nfe"], 1)
        self.assertGreaterEqual(diffusion_stats["diffusion_accepted_tokens"], 1)
        self.assertIn("diffusion_tokens_per_denoise_forward", diffusion_stats)

        for sampler in (
            "native",
            "fixed",
            "confidence_threshold_ref",
            "confidence_threshold_bound",
            "cumulative_error",
        ):
            with self.subTest(sampler=sampler):
                sampled = model.language_model.generate(
                    mx.array([[4]], dtype=mx.int32),
                    block_length=2,
                    steps=2,
                    gen_length=2,
                    mask_id=127,
                    eos_id=999,
                    sampler=sampler,
                    threshold=0.5,
                )
                self.assertEqual(sampled.shape, (1, 2))

        with self.assertRaises(ValueError):
            model.language_model.generate(
                mx.array([[4]], dtype=mx.int32),
                block_length=2,
                gen_length=2,
                mask_id=127,
                eos_id=999,
                sampler="bogus",
            )

        mixed = model.language_model.generate(
            mx.array([[4]], dtype=mx.int32),
            block_length=2,
            gen_length=2,
            mask_id=127,
            eos_id=999,
            ar_weight=0.5,
        )
        self.assertEqual(mixed.shape, (1, 2))

        with self.assertRaises(ValueError):
            model.language_model.generate(
                mx.array([[4]], dtype=mx.int32),
                block_length=2,
                gen_length=2,
                mask_id=127,
                eos_id=999,
                ar_weight=1.5,
            )

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
            threshold=0.5,
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
                sampler="native",
                sampling_scaling_factor=2.0,
                head_scoring="chunked",
                temperature=0.0,
            )
        )
        self.assertTrue(diffusion_calls["kwargs"])
        self.assertNotIn("linear_speculative", diffusion_calls["kwargs"])
        self.assertNotIn("steps", diffusion_calls["kwargs"])
        self.assertNotIn("threshold", diffusion_calls["kwargs"])
        self.assertEqual(diffusion_calls["kwargs"]["sampler"], "native")
        self.assertEqual(diffusion_calls["kwargs"]["sampling_scaling_factor"], 2.0)
        self.assertEqual(diffusion_calls["kwargs"]["head_scoring"], "chunked")
        self.assertEqual(diffusion_results[-1].generation_tokens, 2)

        diffusion_calls.clear()
        list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4]], dtype=mx.int32),
                max_tokens=2,
                generation_mode="dlm",
                temperature=0.0,
            )
        )
        self.assertTrue(diffusion_calls["kwargs"])
        self.assertNotIn("linear_speculative", diffusion_calls["kwargs"])

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
                linear_speculative=True,
                temperature=0.0,
            )
        )
        self.assertTrue(linear_calls["kwargs"])
        self.assertTrue(linear_calls["kwargs"]["linear_speculative"])
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[-1].generation_tokens, 2)
        self.assertEqual(results[-1].finish_reason, "stop")


class TestMaskedDiffusionServerLane(unittest.TestCase):
    def _tiny_llada(self):
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
        return llada2_moe.Model(config)

    def test_generate_invokes_on_block_per_block(self):
        mx.random.seed(0)
        model = self._tiny_llada()
        blocks = []

        def on_block(tokens):
            blocks.append(list(tokens))
            return True

        generated = model.language_model.generate(
            mx.array([[4, 5]], dtype=mx.int32),
            gen_length=8,
            block_length=4,
            steps=4,
            eos_early_stop=False,
            on_block=on_block,
        )

        self.assertGreaterEqual(len(blocks), 2)
        # Each callback reports the cumulative generated tokens so far.
        self.assertLess(len(blocks[0]), len(blocks[-1]))
        self.assertEqual(blocks[-1][: len(blocks[0])][:0], [])
        self.assertLessEqual(len(blocks[-1]), 8)
        self.assertEqual(generated.shape[0], 1)

    def test_generate_on_block_false_stops_early(self):
        mx.random.seed(0)
        model = self._tiny_llada()
        calls = []

        def on_block(tokens):
            calls.append(len(tokens))
            return False

        model.language_model.generate(
            mx.array([[4, 5]], dtype=mx.int32),
            gen_length=8,
            block_length=4,
            steps=4,
            eos_early_stop=False,
            on_block=on_block,
        )

        self.assertEqual(len(calls), 1)

    def test_llada_unmasking_visualizes_current_block(self):
        from mlx_vlm.models.llada2_moe import language as llada_language

        mx.random.seed(0)
        model = self._tiny_llada()
        calls = []
        original_visualizer = llada_language.DiffusionUnmaskingVisualizer

        class FakeVisualizer:
            def __init__(self, **kwargs):
                self.active = True

            def visualize(self, tokens, force=False):
                calls.append((tokens.shape[1], force))

            def finish(self):
                pass

        llada_language.DiffusionUnmaskingVisualizer = FakeVisualizer
        try:
            model.language_model.generate(
                mx.array([[4, 5, 6, 7]], dtype=mx.int32),
                gen_length=8,
                block_length=4,
                steps=1,
                eos_early_stop=False,
                visualize=True,
                mask_id=127,
                eos_id=999,
            )
        finally:
            llada_language.DiffusionUnmaskingVisualizer = original_visualizer

        force_lengths = [length for length, force in calls if force]
        self.assertEqual(force_lengths, [4, 8])
        self.assertEqual(calls[0], (4, True))

    def test_unmasking_visualizer_preserves_decoded_newlines(self):
        from mlx_vlm.models.diffusion_visualizer import DiffusionUnmaskingVisualizer

        class NewlineTokenizer:
            def decode(self, tokens, skip_special_tokens=False):
                token = int(tokens[0])
                if token == 5:
                    return "\n"
                return str(token)

        visualizer = DiffusionUnmaskingVisualizer(
            active=True,
            mask_id=127,
            eos_token_ids=[],
            tokenizer=NewlineTokenizer(),
            min_interval=0.0,
        )
        drawn = []

        class FakeRedrawer:
            def throttled(self):
                return False

            def draw(self, text, force=False):
                drawn.append(text)

            def finish(self):
                pass

        visualizer.redrawer = FakeRedrawer()
        visualizer.visualize(mx.array([[4, 5, 6]], dtype=mx.int32), force=True)

        self.assertEqual(drawn[-1], "4\n6")

    def test_dispatch_routes_diffusion_models_through_shared_adapter(self):
        from mlx_vlm.generate.common import GenerationResult
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig
        from mlx_vlm.tests.test_diffusion_gemma import tiny_config_dict

        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
        models = [
            self._tiny_llada(),
            Model(ModelConfig.from_dict(tiny_config_dict())),
        ]
        original = dispatch.stream_diffusion_generate_from_kwargs
        calls = []

        def fake_stream(model, *args, **kwargs):
            calls.append(model)
            yield GenerationResult(text="ok", generation_tokens=1)

        dispatch.stream_diffusion_generate_from_kwargs = fake_stream
        try:
            for model in models:
                results = list(
                    stream_generate(
                        model,
                        _Processor(),
                        prompt="ignored",
                        input_ids=mx.array([[4]], dtype=mx.int32),
                        max_tokens=1,
                        temperature=0.0,
                    )
                )
                self.assertEqual(results[-1].text, "ok")
        finally:
            dispatch.stream_diffusion_generate_from_kwargs = original

        self.assertEqual(calls, models)

    def test_diffusion_generation_family_routing(self):
        from mlx_vlm.generate.diffusion import (
            diffusion_generation_family,
            is_block_diffusion_model,
            is_diffusion_model,
            is_masked_diffusion_model,
        )

        model = self._tiny_llada()
        self.assertTrue(is_diffusion_model(model))
        self.assertTrue(is_masked_diffusion_model(model))
        self.assertFalse(is_block_diffusion_model(model))
        self.assertEqual(diffusion_generation_family(model), "masked")

        # Mask-token models that default to AR stay on the batch generator.
        model.config.default_generation_mode = "ar"
        self.assertIsNone(diffusion_generation_family(model))
        model.config.default_generation_mode = None

        model.config.mask_token_id = None
        self.assertIsNone(diffusion_generation_family(model))

    def test_diffusion_generation_family_block(self):
        from mlx_vlm.generate.diffusion import (
            diffusion_generation_family,
            is_block_diffusion_model,
            is_diffusion_model,
            is_masked_diffusion_model,
        )
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig
        from mlx_vlm.tests.test_diffusion_gemma import tiny_config_dict

        model = Model(ModelConfig.from_dict(tiny_config_dict()))
        self.assertTrue(is_diffusion_model(model))
        self.assertTrue(is_block_diffusion_model(model))
        self.assertFalse(is_masked_diffusion_model(model))
        self.assertEqual(diffusion_generation_family(model), "block")

    def test_block_diffusion_requires_backend_marker(self):
        from mlx_vlm.generate.diffusion import is_block_diffusion_model

        class Config:
            canvas_length = 4

        class Model:
            config = Config()

            def diffusion_decoder_logits(self):
                pass

            def diffusion_decoder_masks(self):
                pass

            def diffusion_prefill_cache(self):
                pass

            def diffusion_prepare_self_conditioning(self):
                pass

            def diffusion_self_conditioning(self):
                pass

            def diffusion_update_cache(self):
                pass

        self.assertFalse(is_block_diffusion_model(Model()))


if __name__ == "__main__":
    unittest.main()
