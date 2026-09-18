"""Text diffusion model, generation, numerical parity, and vision contracts."""

from __future__ import annotations

import json
import math
import unittest
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_vlm.generate import diffusion
from mlx_vlm.generate.common import GenerationResult
from mlx_vlm.generate.dispatch import stream_generate
from mlx_vlm.models.cache import StaticPrefixKVCache
from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
from mlx_vlm.utils import StoppingCriteria, load_config

# Text diffusion model, generation, numerical parity, and vision contracts.


def tiny_config_dict():
    return {
        "model_type": "diffusion_gemma",
        "canvas_length": 3,
        "image_token_id": 258880,
        "text_config": {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_experts": 4,
            "top_k_experts": 2,
            "use_bidirectional_attention": None,
            "final_logit_softcapping": 30.0,
        },
        "vision_config": None,
        "generation_config": {
            "max_denoising_steps": 1,
            "sampler_config": {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": 0.1,
            },
            "linear_temperature_schedule_config": {
                "_cls_name": "LinearTemperatureScheduleConfig",
                "t_min": 0.4,
                "t_max": 0.8,
            },
        },
    }


def tiny_vision_config_dict():
    config = tiny_config_dict()
    config["image_token_id"] = 60
    config["video_token_id"] = 61
    config["text_config"]["use_bidirectional_attention"] = "vision"
    config["vision_config"] = {
        "model_type": "gemma4_vision",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "patch_size": 2,
        "pooling_kernel_size": 2,
        "default_output_length": 1,
        "position_embedding_size": 8,
    }
    return config


class FakeTokenizer:
    all_special_ids = []
    eos_token_ids = [999999]

    def __init__(self):
        self.stopping_criteria = StoppingCriteria([999999], self)

    def decode(self, tokens, **kwargs):
        return "".join(chr(65 + (int(token) % 26)) for token in tokens)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.detokenizer = NaiveStreamingDetokenizer(self.tokenizer)


class RecordingEncoder:
    def __init__(self, inner):
        self.inner = inner
        self.input_lengths = []
        self.attention_masks = []
        self.mm_token_type_ids = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, input_ids, *args, **kwargs):
        self.input_lengths.append(input_ids.shape[1])
        self.attention_masks.append(kwargs.get("attention_mask"))
        self.mm_token_type_ids.append(kwargs.get("mm_token_type_ids"))
        return self.inner(input_ids, *args, **kwargs)


def make_diffusion_model(config=None, *, vision=False, seed=0):
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

    mx.random.seed(seed)
    if config is None:
        config = tiny_vision_config_dict() if vision else tiny_config_dict()
    return Model(ModelConfig.from_dict(config))


def diffusion_responses(model, input_ids=(2, 3), **kwargs):
    from mlx_vlm.generate import stream_generate

    options = {"max_tokens": 2, **kwargs}
    return list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([input_ids], dtype=mx.int32),
            **options,
        )
    )


def _llada_config(**overrides):
    """Build a fresh config; callers specify only scenario-specific differences."""
    from mlx_vlm.models import llada2_moe

    fields = dict(
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
    fields.update(overrides)
    return llada2_moe.ModelConfig(**fields)


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


def _native_stream(model, **options):
    options = (
        dict(
            prompt="ignored",
            input_ids=mx.array([[4]], mx.int32),
            max_tokens=2,
            temperature=0.0,
        )
        | options
    )
    return stream_generate(model, _Processor(), **options)


def _native_generate(model, **options):
    options = dict(block_length=2, gen_length=2, mask_id=127, eos_id=999) | options
    return model.language_model.generate(mx.array([[4]], mx.int32), **options)


def _record_generation(model, tokens):
    calls = {}

    def generate(input_ids, **kwargs):
        calls.update(kwargs)
        kwargs["stats"]["prompt_time"] = 1.0
        kwargs["on_result"](
            GenerationResult(
                text="decoded",
                token=3,
                prompt_tokens=input_ids.size,
                generation_tokens=len(tokens),
                total_tokens=input_ids.size + len(tokens),
                finish_reason="stop",
            )
        )
        return mx.array([tokens], mx.int32)

    model.language_model.generate = generate
    return calls


class TestDiffusionModels(unittest.TestCase):
    def dtype_consistency_test_runner(self, language_model, model_type, num_layers):
        assert language_model.model_type == model_type
        assert len(language_model.layers) == num_layers
        inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

        for dtype in [mx.float32, mx.float16]:
            with self.subTest(dtype=dtype):
                language_model.update(
                    tree_map(lambda p: p.astype(dtype), language_model.parameters())
                )

                outputs = language_model(inputs)
                assert outputs.logits.dtype == dtype

                prefix_cache = [
                    StaticPrefixKVCache(max_size=8) for _ in language_model.layers
                ]
                mx.eval(language_model.model(inputs[:, :2], cache=prefix_cache))
                block_cache = [StaticPrefixKVCache.from_prefix(c) for c in prefix_cache]
                cached_outputs = language_model(inputs[:, 2:], cache=block_cache)
                assert cached_outputs.logits.dtype == dtype

    def test_llada(self):
        from mlx_vlm.models import llada2_moe

        config = _llada_config()
        model = llada2_moe.Model(config)
        captured = _record_generation(model, [1, 2, 3])
        tuned = dict(
            max_denoising_steps=7,
            block_length=16,
            num_to_transfer=3,
            max_transfer_per_step=2,
            threshold=0.8,
            min_threshold=0.6,
            editing_threshold=0.7,
            max_post_steps=2,
            stability_steps=1,
        )
        for options in ({}, tuned, dict(num_to_transfer=2)):
            captured.clear()
            result = next(
                _native_stream(
                    model,
                    input_ids=mx.array([[4, 5]], mx.int32),
                    max_tokens=8,
                    **options,
                )
            )
            assert result.text == "decoded" and result.generation_tokens == 3
            expected = {
                ("steps" if key == "max_denoising_steps" else key): value
                for key, value in options.items()
            }
            if not options:
                expected = dict(block_length=32, steps=32)
                assert (
                    not (set(tuned) - {"max_denoising_steps", "block_length"})
                    & captured.keys()
                )
            elif options != tuned:
                assert (
                    "threshold" not in captured and "editing_threshold" not in captured
                )
            assert {key: captured[key] for key in expected} == expected
        model = llada2_moe.Model(config)
        self.dtype_consistency_test_runner(
            model.language_model, config.model_type, config.num_hidden_layers
        )

    def test_llada_stream_generate_ignores_extra_cli_kwargs(self):
        from mlx_vlm.models import llada2_moe

        config = _llada_config()
        model = llada2_moe.Model(config)

        results = list(_native_stream(model, steps=1, fps=2.0))

        assert results[-1].generation_tokens == 2

        with self.assertRaisesRegex(ValueError, "does not support linear_speculative"):
            list(
                _native_stream(
                    model, generation_mode="diffusion", linear_speculative=True
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
            model.language_model, config.model_type, config.num_hidden_layers
        )

        model.language_model.update(
            tree_map(lambda p: p.astype(mx.bfloat16), model.language_model.parameters())
        )
        bf16_outputs = model.language_model(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            attention_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
        )
        assert bf16_outputs.logits.dtype == mx.bfloat16
        bf16_filtered = model.language_model._top_k_logits(bf16_outputs.logits, 2)
        assert bf16_filtered.dtype == mx.bfloat16
        _, bf16_probs = model.language_model._sample_with_temperature_topk_topp(
            bf16_outputs.logits
        )
        assert bf16_probs.dtype == mx.bfloat16

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
        assert score_tokens.tolist() == ref_tokens.tolist()
        assert bool(mx.allclose(score_probs, ref_probs).item())

        diffusion_stats = {}
        generated = _native_generate(
            model,
            block_length=4,
            steps=1,
            gen_length=8,
            max_post_steps=4,
            stats=diffusion_stats,
        )
        assert generated.shape == (1, 8)
        assert diffusion_stats["diffusion_sampler"] == "native"
        assert math.isnan(diffusion_stats["diffusion_min_threshold"])
        assert diffusion_stats["diffusion_transformers_parity"] == 1.0
        assert diffusion_stats["diffusion_denoise_nfe"] >= 1
        assert diffusion_stats["diffusion_accepted_tokens"] >= 1
        assert "diffusion_tokens_per_denoise_forward" in diffusion_stats

        for sampler in (
            "native",
            "fixed",
            "confidence_threshold_ref",
            "confidence_threshold_bound",
            "cumulative_error",
        ):
            with self.subTest(sampler=sampler):
                sampled = _native_generate(
                    model, steps=2, sampler=sampler, threshold=0.5
                )
                assert sampled.shape == (1, 2)

        with self.assertRaises(ValueError):
            _native_generate(model, sampler="bogus")

        mixed = _native_generate(model, ar_weight=0.5)
        assert mixed.shape == (1, 2)

        with self.assertRaises(ValueError):
            _native_generate(model, ar_weight=1.5)

        for method, options in [
            (model.language_model.ar_generate, {}),
            (
                model.language_model.linear_spec_generate,
                dict(block_length=2, mask_token_id=127, threshold=0.5),
            ),
        ]:
            tokens, nfe = method(
                mx.array([[4]], mx.int32), max_new_tokens=2, eos_token_id=3, **options
            )
            mx.eval(tokens)
            assert tokens.shape[0] == 1 and tokens.shape[1] <= 3 and nfe >= 1

        spec_results = list(
            _native_stream(
                model, generation_mode="linear_speculative", linear_speculative=True
            )
        )
        assert len(spec_results) >= 1
        assert spec_results[-1].generation_tokens == 2

        def unexpected_diffusion_generate(*args, **kwargs):
            raise AssertionError("Default Nemotron generation should use AR")

        model.language_model.generate = unexpected_diffusion_generate
        default_results = list(_native_stream(model, max_tokens=1))
        assert default_results[-1].generation_tokens == 1

        captured = _record_generation(model, [5, 3])
        for mode, options in [
            (
                "diffusion",
                dict(
                    sampler="native",
                    sampling_scaling_factor=2.0,
                    head_scoring="chunked",
                ),
            ),
            ("dlm", {}),
            ("linear_speculative", dict(linear_speculative=True)),
        ]:
            captured.clear()
            results = list(_native_stream(model, generation_mode=mode, **options))
            assert captured
            if mode == "linear_speculative":
                assert captured["linear_speculative"]
                assert len(results) >= 1 and results[-1].finish_reason == "stop"
            else:
                assert "linear_speculative" not in captured
            if mode == "diffusion":
                expected = dict(steps=32, threshold=0.9, **options)
                assert {key: captured[key] for key in expected} == expected
            if mode != "dlm":
                assert results[-1].generation_tokens == 2


class TestMaskedDiffusionServerLane(unittest.TestCase):
    def _tiny_llada(self):
        from mlx_vlm.models import llada2_moe

        config = _llada_config()
        return llada2_moe.Model(config)

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

        assert len(calls) == 1

    def test_stream_generate_yields_llada_model_owned_results(self):
        mx.random.seed(0)
        model = self._tiny_llada()

        responses = list(
            stream_generate(
                model,
                _Processor(),
                prompt="ignored",
                input_ids=mx.array([[4, 5]], dtype=mx.int32),
                max_tokens=8,
                max_denoising_steps=4,
                block_length=4,
                temperature=0.0,
            )
        )

        assert any((result.diffusion_block_complete for result in responses))
        assert not responses[-1].diffusion_block_complete
        assert responses[-1].finish_reason is not None

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
        assert force_lengths == [4, 8]
        assert calls[0] == (4, True)

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

        assert drawn[-1] == "4\n6"

    def test_diffusion_generation_family_routing(self):
        from mlx_vlm.generate.diffusion import (
            diffusion_generation_family,
            is_diffusion_model,
        )

        model = self._tiny_llada()
        assert is_diffusion_model(model)
        assert diffusion_generation_family(model) == "diffusion"

        # Mask-token models that default to AR stay on the batch generator.
        model.config.default_generation_mode = "ar"
        assert not is_diffusion_model(model)
        assert diffusion_generation_family(model) is None
        assert is_diffusion_model(model, {"generation_mode": "diffusion"})
        assert (
            diffusion_generation_family(model, {"generation_mode": "diffusion"})
            == "diffusion"
        )
        model.config.default_generation_mode = None

        model.config.mask_token_id = None
        assert not is_diffusion_model(model)
        assert diffusion_generation_family(model) is None


class TestDiffusionGemma:
    def test_chunked_diffusion_prefill_matches_unchunked_tokens(self):
        def tokens(step_size):
            responses = diffusion_responses(
                make_diffusion_model(seed=42),
                (2, 3, 4, 5, 6),
                max_tokens=3,
                prefill_step_size=step_size,
            )
            return [r.token for r in responses if r.token is not None]

        assert tokens(None) == tokens(2)

    @pytest.mark.parametrize("visual", [False, True], ids=["padded", "visual"])
    def test_diffusion_prefill_stays_unchunked(self, visual):
        model = make_diffusion_model()
        recorder = RecordingEncoder(model.model.encoder)
        model.model.encoder = recorder
        metadata = mx.array(
            [[0, 1, 1, 0, 0]] if visual else [[1, 1, 1, 1, 0]],
            dtype=mx.int32 if visual else mx.bool_,
        )
        kwargs = {"mm_token_type_ids" if visual else "mask": metadata}
        responses = diffusion_responses(
            model,
            (2, 3, 4, 5, 6 if visual else 0),
            max_tokens=1,
            prefill_step_size=2,
            **kwargs,
        )
        assert responses[-1].generation_tokens == 1
        assert recorder.input_lengths == [5]
        if visual:
            assert recorder.mm_token_type_ids[0] is metadata
        else:
            assert recorder.attention_masks[0] is not None

    def test_stream_generate_supports_static_diffusion_cache(self):
        result = diffusion_responses(
            make_diffusion_model(), max_tokens=4, diffusion_static_cache=True
        )[-1]
        assert result.generation_tokens == 4
        assert result.diffusion_canvas_tokens >= 3

    def test_default_confidence_threshold_sampler_can_exit_after_one_step(self):
        result = diffusion_responses(
            make_diffusion_model(), max_denoising_steps=4, diffusion_threshold=0.0
        )[-1]
        assert result.generation_tokens == 2
        assert result.diffusion_denoising_steps == 1
        assert result.diffusion_work_tokens == 3

    def test_stream_generate_uses_checkpoint_denoising_steps(self):
        config = tiny_config_dict()
        config["generation_config"]["max_denoising_steps"] = 48
        result = diffusion_responses(
            make_diffusion_model(config), diffusion_sampler="entropy-bound"
        )[-1]
        assert result.diffusion_denoising_steps == 48
        assert result.diffusion_work_tokens == 48 * 3

    def test_diffusion_initial_canvas_pads_short_decoder_input_ids(self):
        decoder_ids = diffusion._normalize_decoder_input_ids(
            [[10, 11]], batch_size=1, dtype=mx.int32
        )
        with patch.object(
            diffusion,
            "_diffusion_initialize_canvas",
            return_value=mx.array([[7, 8, 9]], dtype=mx.int32),
        ):
            canvas = diffusion._diffusion_initial_canvas(
                decoder_ids,
                start_index=0,
                batch_size=1,
                canvas_length=3,
                vocab_size=64,
                dtype=mx.int32,
            )
            mx.eval(canvas)
        assert canvas.tolist() == [[10, 11, 9]]

    @pytest.mark.parametrize(
        "ids,error", [([10, 11], "2D array"), ([[10, 11], [12, 13]], "batch size")]
    )
    def test_diffusion_decoder_input_ids_validation(self, ids, error):
        with pytest.raises(ValueError, match=error):
            diffusion._normalize_decoder_input_ids(ids, batch_size=1, dtype=mx.int32)

    def test_stream_generate_slices_decoder_input_ids_by_canvas(self):
        model = make_diffusion_model()
        seen = []

        def logits(canvas, *args, **kwargs):
            mx.eval(canvas)
            seen.append(canvas.tolist())
            return mx.zeros((*canvas.shape, model.config.text_config.vocab_size))

        with patch.object(model, "diffusion_decoder_logits", side_effect=logits):
            responses = diffusion_responses(
                model,
                max_tokens=4,
                diffusion_max_canvas_length=2,
                decoder_input_ids=mx.array([[10, 11, 12, 13]], dtype=mx.int32),
            )
        assert responses[-1].generation_tokens == 4
        assert seen[:2] == [[[10, 11]], [[12, 13]]]

    @pytest.mark.parametrize(
        "temperature,expected",
        [(0.0, [[1, 0]]), (0.7, [[2, 1]])],
        ids=["argmax", "categorical"],
    )
    def test_diffusion_samples_canvas(self, temperature, expected):
        logits = mx.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]])
        with patch.object(
            mx.random, "categorical", return_value=mx.array([[2, 1]])
        ) as categorical:
            sampled = diffusion._diffusion_sample_canvas(
                logits, mx.int32, temperature=temperature
            )
            mx.eval(sampled)
        assert categorical.call_count == (1 if temperature else 0)
        assert sampled.tolist() == expected

    def test_stream_generate_with_mxfp4_quantized_embeddings(self):
        import mlx.nn as nn

        config = tiny_config_dict()
        config["text_config"]["hidden_size"] = 32
        config["generation_config"]["max_denoising_steps"] = 3
        model = make_diffusion_model(config)
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            mode="mxfp4",
            class_predicate=lambda path, module: isinstance(module, nn.Embedding),
        )
        assert model.model.decoder.embed_tokens.mode == "mxfp4"
        result = diffusion_responses(model)[-1]
        assert result.generation_tokens == 2
        assert result.diffusion_work_tokens > 0

    def test_precomputed_self_conditioning_embeddings_match_logits_path(self):
        model = make_diffusion_model()
        config = model.config
        decoder = model.model.decoder
        decoder.embed_tokens.weight = decoder.embed_tokens.weight.astype(mx.bfloat16)
        input_ids = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
        canvas_ids = mx.array([[6, 7, 8]], dtype=mx.int32)
        self_conditioning_logits = mx.linspace(
            -0.5, 0.5, config.text_config.vocab_size * canvas_ids.shape[-1]
        ).reshape(1, canvas_ids.shape[-1], -1)
        stored_self_conditioning_logits = self_conditioning_logits.astype(
            decoder.embed_tokens.weight.dtype
        )

        self_conditioning_embeddings = model.diffusion_self_conditioning(
            self_conditioning_logits, model.diffusion_prepare_self_conditioning()
        ).astype(decoder.embed_tokens.weight.dtype)
        logits_output = model(
            input_ids=input_ids,
            canvas_ids=canvas_ids,
            self_conditioning_logits=stored_self_conditioning_logits,
        ).logits
        embeddings_output = model(
            input_ids=input_ids,
            canvas_ids=canvas_ids,
            self_conditioning_embeddings=self_conditioning_embeddings,
        ).logits
        max_diff = mx.max(mx.abs(logits_output - embeddings_output))
        mx.eval(max_diff)

        assert float(max_diff.item()) < 1e-05

    def test_transformers_58_logits_and_denoising_step_parity_if_available(self):
        try:
            import numpy as np
            import torch
            from transformers.cache_utils import DynamicCache
            from transformers.generation.logits_process import LogitsProcessorList
            from transformers.models.diffusion_gemma4.generation_diffusion_gemma4 import (
                LinearTemperatureScheduleConfig,
                LinearTemperatureScheduleLogitsProcessor,
            )
            from transformers.models.diffusion_gemma4.modeling_diffusion_gemma4 import (
                DiffusionGemma4Config,
                DiffusionGemma4ModelForBlockDiffusion,
            )
        except Exception as exc:
            pytest.skip(
                f"Transformers 5.8 DiffusionGemma4 reference unavailable: {exc}"
            )

        from mlx_vlm.generate.diffusion import _diffusion_linear_temperature
        from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

        class ArgmaxNoRenoiseSampler:
            def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
                return torch.argmax(logits, dim=-1)

            def renoise_canvas(self, accepted_canvas, cur_step):
                return accepted_canvas

        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 4
        torch.manual_seed(123)
        hf_model = DiffusionGemma4ModelForBlockDiffusion(
            DiffusionGemma4Config(**config_dict)
        ).eval()
        mlx_model = Model(ModelConfig.from_dict(config_dict))
        weights = {
            key: mx.array(value.detach().cpu().numpy())
            for key, value in hf_model.state_dict().items()
        }
        mlx_model.load_weights(list(mlx_model.sanitize(weights).items()), strict=False)

        input_ids_t = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        canvas_t = torch.tensor([[6, 7, 8]], dtype=torch.long)
        input_ids_m = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
        canvas_m = mx.array([[6, 7, 8]], dtype=mx.int32)

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids_t, canvas_ids=canvas_t).logits
        mlx_logits = mlx_model(input_ids=input_ids_m, canvas_ids=canvas_m).logits
        mx.eval(mlx_logits)
        assert (
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            )
            < 1e-05
        )

        sc_logits = np.linspace(
            -0.5,
            0.5,
            canvas_t.numel() * config_dict["text_config"]["vocab_size"],
            dtype=np.float32,
        ).reshape(1, canvas_t.shape[-1], -1)
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=input_ids_t,
                canvas_ids=canvas_t,
                self_conditioning_logits=torch.tensor(sc_logits),
            ).logits
        mlx_logits = mlx_model(
            input_ids=input_ids_m,
            canvas_ids=canvas_m,
            self_conditioning_logits=mx.array(sc_logits),
        ).logits
        mx.eval(mlx_logits)
        assert (
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            )
            < 1e-05
        )

        attention_t = torch.ones_like(input_ids_t, dtype=torch.bool)
        decoder_attention_t = torch.nn.functional.pad(
            attention_t, (0, canvas_t.shape[-1]), value=True
        )
        with torch.no_grad():
            past_key_values = DynamicCache(
                config=hf_model.config.get_text_config(decoder=True)
            )
            encoder_outputs = hf_model.model.encoder(
                input_ids=input_ids_t,
                attention_mask=attention_t,
                past_key_values=past_key_values,
            )
            past_key_values = encoder_outputs.past_key_values
            mask_mapping = (
                hf_model.model.decoder.create_diffusion_decoder_attention_mask(
                    config=hf_model.config.text_config,
                    inputs_embeds=canvas_t.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=decoder_attention_t,
                )
            )
            logits_processor = LogitsProcessorList(
                [
                    LinearTemperatureScheduleLogitsProcessor(
                        LinearTemperatureScheduleConfig(t_min=0.4, t_max=0.8), 4
                    )
                ]
            )
            hf_current, hf_argmax, hf_processed, _ = hf_model._denoising_step(
                decoder_forward=hf_model.forward,
                current_canvas=canvas_t,
                argmax_canvas=canvas_t,
                input_ids=input_ids_t,
                self_conditioning_logits=None,
                mask_mapping=mask_mapping,
                past_key_values=past_key_values,
                finished_denoising=torch.zeros(1, dtype=torch.bool),
                cur_step=3,
                sampler=ArgmaxNoRenoiseSampler(),
                logits_processor=logits_processor,
                diffusion_stopping_criteria=None,
            )

        attention_m = mx.ones(input_ids_m.shape, dtype=mx.bool_)
        kv_cache = mlx_model.make_cache()
        _, kv_cache = mlx_model.model.encoder(
            input_ids_m, attention_mask=attention_m, cache=kv_cache
        )
        decoder_attention_m = mx.concatenate(
            [attention_m, mx.ones(canvas_m.shape, dtype=mx.bool_)], axis=-1
        )
        mask_mapping = mlx_model.model.decoder._make_decoder_masks(
            canvas_m[..., None], kv_cache, decoder_attention_m
        )
        mlx_processed = mlx_model(
            cache=kv_cache, canvas_ids=canvas_m, decoder_attention_mask=mask_mapping
        ).logits / _diffusion_linear_temperature(3, 4, {"t_min": 0.4, "t_max": 0.8})
        mlx_argmax = mx.argmax(mlx_processed, axis=-1).astype(mx.int32)
        mx.eval(mlx_processed, mlx_argmax)

        assert (
            float(
                np.max(
                    np.abs(
                        hf_processed.detach().cpu().numpy() - np.array(mlx_processed)
                    )
                )
            )
            < 1e-05
        )
        assert hf_argmax.detach().cpu().numpy().tolist() == mlx_argmax.tolist()
        assert hf_current.detach().cpu().numpy().tolist() == mlx_argmax.tolist()

    def test_sanitize_maps_fused_experts_and_keeps_encoder_scalars(self):
        model = make_diffusion_model()
        gate_up = mx.zeros((4, 16, 16))
        weights = {
            "model.decoder.layers.0.experts.gate_up_proj": gate_up,
            "model.decoder.layers.0.experts.down_proj": mx.zeros((4, 16, 8)),
            "model.encoder.language_model.layers.0.layer_scalar": mx.ones((1,)),
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (16, 16)
            ),
            "model.encoder.embed_vision.embedding_projection.weight": mx.zeros(
                (16, 16)
            ),
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight": mx.ones(
                (16,)
            ),
            "lm_head.weight": mx.zeros((64, 16)),
        }

        sanitized = model.sanitize(weights)

        assert "model.decoder.layers.0.experts.gate_up_proj.weight" in sanitized
        assert "model.decoder.layers.0.experts.down_proj.weight" in sanitized
        assert sanitized[
            "model.decoder.layers.0.experts.gate_up_proj.weight"
        ].shape == (4, 16, 16)
        assert "model.encoder.language_model.layers.0.layer_scalar" in sanitized
        assert (
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight"
            not in sanitized
        )
        assert "model.encoder.embed_vision.embedding_projection.weight" not in sanitized
        assert (
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight"
            not in sanitized
        )
        assert "lm_head.weight" not in sanitized

    def test_quant_predicate_uses_8bit_for_embeddings_and_attention(self):
        model = make_diffusion_model()
        predicate = model.quant_predicate
        decoder = model.model.decoder

        assert predicate("model.decoder.embed_tokens", decoder.embed_tokens) == {
            "group_size": 64,
            "bits": 8,
        }
        assert predicate(
            "model.decoder.layers.0.self_attn.q_proj",
            decoder.layers[0].self_attn.q_proj,
        ) == {"group_size": 64, "bits": 8}
        assert predicate(
            "model.decoder.layers.0.router.proj", decoder.layers[0].router.proj
        ) == {"group_size": 64, "bits": 8}
        assert predicate(
            "model.decoder.layers.0.mlp.gate_proj", decoder.layers[0].mlp.gate_proj
        ) == {"group_size": 64, "bits": 8}
        assert (
            predicate(
                "model.decoder.layers.0.experts.gate_up_proj",
                decoder.layers[0].experts.gate_up_proj,
            )
            is True
        )

    def test_vision_block_bidirectional_encoder_mask(self):
        model = make_diffusion_model(vision=True)
        config = model.config
        encoder = model.model.encoder

        # text, image, image, text
        mm_token_type_ids = mx.array([[0, 1, 1, 0]])
        h = mx.zeros((1, 4, config.text_config.hidden_size))
        cache = encoder.make_cache()
        masks = encoder._make_encoder_masks(
            h, cache, None, mm_token_type_ids=mm_token_type_ids
        )

        for mask in masks:
            assert mask.shape == (1, 1, 4, 4)
            # Image tokens attend bidirectionally within the block.
            assert bool(mask[0, 0, 1, 2].item())
            # Text tokens stay causal.
            assert not bool(mask[0, 0, 0, 1].item())
            assert not bool(mask[0, 0, 0, 3].item())
            # Later text token sees the whole prefix causally.
            assert bool(mask[0, 0, 3, 0].item())

        # Without vision tokens the fast path is preserved.
        text_masks = encoder._make_encoder_masks(
            h, cache, None, mm_token_type_ids=mx.zeros((1, 4), dtype=mx.int32)
        )
        for mask in text_masks:
            assert not (isinstance(mask, mx.array) and mask.shape == (1, 1, 4, 4))

    def test_video_features_scattered_into_embeddings(self):
        model = make_diffusion_model(vision=True)
        config = model.config

        input_ids = mx.array([[2, config.video_token_id, 3]])
        pixel_values = mx.random.uniform(shape=(1, 3, 4, 4))

        text_only = model.get_input_embeddings(input_ids=input_ids).inputs_embeds
        with_video = model.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values
        ).inputs_embeds

        assert with_video.shape == text_only.shape
        assert bool(mx.allclose(with_video[0, 0], text_only[0, 0]).item())
        assert bool(mx.allclose(with_video[0, 2], text_only[0, 2]).item())
        assert not bool(mx.allclose(with_video[0, 1], text_only[0, 1]).item())

        expected = model.model.encoder.get_image_features(pixel_values).astype(
            with_video.dtype
        )
        assert bool(mx.allclose(with_video[0, 1], expected[0, 0], atol=1e-05).item())

    def test_sanitize_handles_vision_weights(self):
        vision_model = make_diffusion_model(vision=True)
        weights = {
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight": mx.zeros(
                (1,)
            ),
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max": mx.zeros(
                (1,)
            ),
            "model.encoder.embed_vision.embedding_projection.weight": mx.zeros((1,)),
        }
        sanitized = vision_model.sanitize(dict(weights))
        assert (
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
            in sanitized
        )
        assert "model.encoder.embed_vision.embedding_projection.weight" in sanitized
        # Clipping calibration tensors are dropped when clipped linears are off.
        assert (
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max"
            not in sanitized
        )

        text_model = make_diffusion_model()
        sanitized = text_model.sanitize(dict(weights))
        assert sanitized == {}


# Loading and utility contracts


def test_diffusion_gemma_load_config_preserves_generation_config(tmp_path):
    from mlx_vlm.models.diffusion_gemma import ModelConfig

    generation_config = {
        "max_denoising_steps": 48,
        "sampler_config": {
            "_cls_name": "EntropyBoundSamplerConfig",
            "entropy_bound": 0.1,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(tiny_config_dict()))
    (tmp_path / "generation_config.json").write_text(json.dumps(generation_config))
    loaded = load_config(tmp_path)
    assert loaded["model_type"] == "diffusion_gemma"
    assert loaded["generation_config"] == generation_config
    assert ModelConfig.from_dict(loaded).generation_config == generation_config
