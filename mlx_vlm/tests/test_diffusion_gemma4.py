import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlx.core as mx

from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
from mlx_vlm.utils import StoppingCriteria


def tiny_config_dict():
    return {
        "model_type": "diffusion_gemma4",
        "canvas_length": 3,
        "image_token_id": 258880,
        "text_config": {
            "model_type": "diffusion_gemma4_text",
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


class FakeTokenizer:
    all_special_ids = []

    def __init__(self):
        self.stopping_criteria = StoppingCriteria([999999], self)

    def decode(self, tokens, **kwargs):
        return "".join(chr(65 + (int(token) % 26)) for token in tokens)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.detokenizer = NaiveStreamingDetokenizer(self.tokenizer)


class TestDiffusionGemma4(unittest.TestCase):
    def test_model_resolves_from_config(self):
        from mlx_vlm.utils import get_model_and_args

        arch, model_type = get_model_and_args(tiny_config_dict())

        self.assertEqual(model_type, "diffusion_gemma4")
        self.assertEqual(arch.Model.__name__, "Model")

    def test_load_config_normalizes_diffusion_gemma_alias(self):
        from mlx_vlm.utils import load_config

        config = tiny_config_dict()
        config["model_type"] = "diffusion_gemma"
        config["text_config"]["model_type"] = "diffusion_gemma_text"
        generation_config = {
            "max_denoising_steps": 48,
            "sampler_config": {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": 0.1,
            },
        }

        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(config))
            Path(tmpdir, "generation_config.json").write_text(
                json.dumps(generation_config)
            )

            loaded = load_config(Path(tmpdir))

        self.assertEqual(loaded["model_type"], "diffusion_gemma4")
        self.assertEqual(loaded["text_config"]["model_type"], "diffusion_gemma4_text")
        self.assertEqual(loaded["generation_config"], generation_config)

    def test_forward_shape(self):
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)

        out = model(
            input_ids=mx.array([[2, 3, 4]]),
            canvas_ids=mx.array([[5, 6, 7]]),
        )
        mx.eval(out.logits)

        self.assertEqual(out.logits.shape, (1, 3, 64))
        self.assertEqual(len(model.make_cache()), 2)

    def test_self_conditioning_soft_embeddings_use_embedding_scale(self):
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        decoder = model.model.decoder
        captured = {}

        class CaptureSelfConditioning:
            def __call__(self, inputs_embeds, self_conditioning_signal):
                captured["self_conditioning_signal"] = self_conditioning_signal
                return inputs_embeds

        decoder.self_conditioning = CaptureSelfConditioning()
        self_conditioning_logits = mx.zeros((1, 1, config.text_config.vocab_size))
        decoder._embed_canvas(
            mx.array([[2]], dtype=mx.int32),
            self_conditioning_logits,
        )

        expected = (
            mx.softmax(self_conditioning_logits, axis=-1, precise=True)
            @ decoder.embed_tokens.weight
        ) * decoder.embed_scale
        max_diff = mx.max(mx.abs(captured["self_conditioning_signal"] - expected))
        mx.eval(max_diff)

        self.assertLess(float(max_diff.item()), 1e-6)

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
            self.skipTest(f"Transformers 5.8 DiffusionGemma4 reference unavailable: {exc}")

        from mlx_vlm.generate.diffusion import _diffusion_linear_temperature
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

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
        mlx_model.load_weights(
            list(mlx_model.sanitize(weights).items()),
            strict=False,
        )

        input_ids_t = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        canvas_t = torch.tensor([[6, 7, 8]], dtype=torch.long)
        input_ids_m = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
        canvas_m = mx.array([[6, 7, 8]], dtype=mx.int32)

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids_t, canvas_ids=canvas_t).logits
        mlx_logits = mlx_model(input_ids=input_ids_m, canvas_ids=canvas_m).logits
        mx.eval(mlx_logits)
        self.assertLess(
            float(np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))),
            1e-5,
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
        self.assertLess(
            float(np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))),
            1e-5,
        )

        attention_t = torch.ones_like(input_ids_t, dtype=torch.bool)
        decoder_attention_t = torch.nn.functional.pad(
            attention_t,
            (0, canvas_t.shape[-1]),
            value=True,
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
            mask_mapping = hf_model.model.decoder.create_diffusion_decoder_attention_mask(
                config=hf_model.config.text_config,
                inputs_embeds=canvas_t.unsqueeze(-1),
                past_key_values=past_key_values,
                attention_mask=decoder_attention_t,
            )
            logits_processor = LogitsProcessorList(
                [
                    LinearTemperatureScheduleLogitsProcessor(
                        LinearTemperatureScheduleConfig(t_min=0.4, t_max=0.8),
                        4,
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
            input_ids_m,
            attention_mask=attention_m,
            cache=kv_cache,
        )
        decoder_attention_m = mx.concatenate(
            [attention_m, mx.ones(canvas_m.shape, dtype=mx.bool_)],
            axis=-1,
        )
        mask_mapping = mlx_model.model.decoder._make_decoder_masks(
            canvas_m[..., None],
            kv_cache,
            decoder_attention_m,
        )
        mlx_processed = mlx_model(
            cache=kv_cache,
            canvas_ids=canvas_m,
            decoder_attention_mask=mask_mapping,
        ).logits / _diffusion_linear_temperature(3, 4, {"t_min": 0.4, "t_max": 0.8})
        mlx_argmax = mx.argmax(mlx_processed, axis=-1).astype(mx.int32)
        mx.eval(mlx_processed, mlx_argmax)

        self.assertLess(
            float(
                np.max(
                    np.abs(hf_processed.detach().cpu().numpy() - np.array(mlx_processed))
                )
            ),
            1e-5,
        )
        self.assertEqual(hf_argmax.detach().cpu().numpy().tolist(), mlx_argmax.tolist())
        self.assertEqual(hf_current.detach().cpu().numpy().tolist(), mlx_argmax.tolist())

    def test_sanitize_splits_experts_and_keeps_encoder_scalars(self):
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
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

        self.assertIn(
            "model.decoder.layers.0.experts.switch_glu.gate_proj.weight",
            sanitized,
        )
        self.assertIn(
            "model.decoder.layers.0.experts.switch_glu.up_proj.weight",
            sanitized,
        )
        self.assertIn(
            "model.decoder.layers.0.experts.switch_glu.down_proj.weight",
            sanitized,
        )
        self.assertEqual(
            sanitized[
                "model.decoder.layers.0.experts.switch_glu.gate_proj.weight"
            ].shape,
            (4, 8, 16),
        )
        self.assertIn("model.encoder.language_model.layers.0.layer_scalar", sanitized)
        self.assertNotIn(
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight",
            sanitized,
        )
        self.assertNotIn(
            "model.encoder.embed_vision.embedding_projection.weight",
            sanitized,
        )
        self.assertNotIn(
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight",
            sanitized,
        )
        self.assertNotIn("lm_head.weight", sanitized)

    def test_stream_generate_uses_diffusion_loop(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=1,
            )
        )

        self.assertGreaterEqual(len(responses), 1)
        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].prompt_tokens, 2)
        self.assertEqual(responses[-1].diffusion_canvas_tokens, 3)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 1)
        self.assertEqual(responses[-1].diffusion_work_tokens, 3)

    def test_decoder_masks_skip_no_padding_short_context(self):
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        cache = model.make_cache()
        _, cache = model.model.encoder(
            mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache
        )

        masks = model.model.decoder._make_decoder_masks(
            mx.zeros((1, 3, 1)),
            cache,
            decoder_attention_mask=None,
        )

        self.assertIsNone(masks["full_attention"])
        self.assertIsNone(masks["sliding_attention"])

    def test_static_prefix_cache_exposes_full_decoder_state(self):
        from mlx_vlm.models.cache import StaticPrefixKVCache

        cache = StaticPrefixKVCache(max_size=5)
        keys = mx.ones((1, 2, 3, 4))
        values = mx.ones((1, 2, 3, 4)) * 2

        prefix_keys, prefix_values = cache.update_and_fetch(keys, values)

        self.assertEqual(prefix_keys.shape, (1, 2, 3, 4))
        self.assertEqual(prefix_values.shape, (1, 2, 3, 4))
        self.assertEqual(cache.decoder_state[0].shape, (1, 2, 5, 4))
        self.assertEqual(cache.offset, 3)

    def test_stream_generate_supports_static_diffusion_cache(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=4,
                max_denoising_steps=1,
                diffusion_static_cache=True,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 4)
        self.assertGreaterEqual(responses[-1].diffusion_canvas_tokens, 3)

    def test_confidence_threshold_sampler_can_exit_after_one_step(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=4,
                diffusion_sampler="confidence-threshold",
                diffusion_threshold=0.0,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 1)
        self.assertEqual(responses[-1].diffusion_work_tokens, 3)

    def test_entropy_bound_sampler_runs_configured_steps(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        mx.random.seed(0)
        config_dict = tiny_config_dict()
        config_dict["generation_config"]["max_denoising_steps"] = 4
        config_dict["generation_config"]["sampler_config"] = {
            "_cls_name": "EntropyBoundSamplerConfig",
            "entropy_bound": 1_000.0,
        }
        config = ModelConfig.from_dict(config_dict)
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
            )
        )

        self.assertEqual(responses[-1].generation_tokens, 2)
        self.assertEqual(responses[-1].diffusion_denoising_steps, 4)
        self.assertEqual(responses[-1].diffusion_work_tokens, 12)

    def test_stream_generate_can_emit_unmasking_drafts(self):
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.models.diffusion_gemma4 import Model, ModelConfig

        mx.random.seed(0)
        config = ModelConfig.from_dict(tiny_config_dict())
        model = Model(config)
        processor = FakeProcessor()

        responses = list(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([[2, 3]], dtype=mx.int32),
                max_tokens=2,
                max_denoising_steps=2,
                diffusion_show_unmasking=True,
            )
        )
        drafts = [response for response in responses if response.is_draft]
        finals = [response for response in responses if not response.is_draft]

        self.assertEqual(len(drafts), 3)
        self.assertTrue(all(response.draft_text for response in drafts))
        self.assertEqual(drafts[0].diffusion_step, 0)
        self.assertIn("[Mask]", drafts[0].draft_text)
        self.assertEqual(drafts[1].diffusion_step, 1)
        self.assertEqual(drafts[-1].diffusion_total_steps, 2)
        self.assertEqual(finals[-1].generation_tokens, 2)

    def test_diffusion_zero_temperature_still_samples_canvas(self):
        from mlx_vlm.generate.diffusion import _diffusion_sample_canvas

        logits = mx.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]])

        with patch("mlx_vlm.generate.mx.random.categorical") as categorical:
            categorical.return_value = mx.array([[2, 1]])
            sampled = _diffusion_sample_canvas(logits, mx.int32, temperature=0.0)
            mx.eval(sampled)

        categorical.assert_called_once()
        self.assertEqual(sampled.tolist(), [[2, 1]])

    def test_diffusion_confidence_transfer_forces_best_unrevealed_token(self):
        from mlx_vlm.generate.diffusion import _diffusion_confidence_transfer_mask

        confidence = mx.array([[0.1, 0.4, 0.2]])
        unrevealed = mx.array([[True, True, False]])

        transfer = _diffusion_confidence_transfer_mask(
            confidence,
            unrevealed,
            threshold=0.9,
        )
        mx.eval(transfer)

        self.assertEqual(transfer.tolist(), [[False, True, False]])

    def test_diffusion_entropy_transfer_forces_best_unrevealed_token(self):
        from mlx_vlm.generate.diffusion import _diffusion_entropy_transfer_mask

        entropy = mx.array([[0.4, 0.1, 0.2, 0.05]])

        transfer = _diffusion_entropy_transfer_mask(
            entropy,
            entropy_bound=0.16,
        )
        mx.eval(transfer)

        self.assertEqual(transfer.tolist(), [[False, True, True, True]])

    def test_unmasking_display_has_no_prefix_or_real_newlines(self):
        from mlx_vlm.generate import GenerationResult
        from mlx_vlm.generate.diffusion import (
            _format_diffusion_draft_line,
            _format_diffusion_live_text,
        )

        draft = GenerationResult(
            is_draft=True,
            draft_text="[Mask] Hello",
            diffusion_canvas_index=1,
            diffusion_step=1,
            diffusion_total_steps=4,
        )

        self.assertEqual(_format_diffusion_draft_line(draft, 80), "[Mask] Hello")
        self.assertEqual(
            _format_diffusion_live_text("hello\nworld", 80), "hello\\nworld"
        )

    def test_unmasking_display_is_untrimmed_by_default(self):
        from mlx_vlm.generate import GenerationResult
        from mlx_vlm.generate.diffusion import (
            _format_diffusion_draft_line,
            _format_diffusion_live_text,
        )

        long_text = "A" * 200
        draft = GenerationResult(is_draft=True, draft_text=long_text)

        self.assertEqual(_format_diffusion_draft_line(draft), long_text)
        self.assertEqual(_format_diffusion_live_text(long_text), long_text)
        self.assertTrue(_format_diffusion_live_text(long_text, 20).endswith("..."))

    def test_generate_redraw_mode_prints_full_final_text(self):
        from mlx_vlm.generate import GenerationResult, generate

        class Config:
            model_type = "diffusion_gemma4"
            eos_token_id = 999999

        class Model:
            config = Config()

        long_text = "The sky is blue because Rayleigh scattering favors blue light."
        chunks = [
            GenerationResult(is_draft=True, draft_text="[Mask] [Mask] [Mask]"),
            GenerationResult(
                text=long_text,
                token=1,
                prompt_tokens=3,
                generation_tokens=12,
                total_tokens=15,
                prompt_tps=10.0,
                generation_tps=5.0,
            ),
        ]

        buffer = io.StringIO()
        with (
            patch("mlx_vlm.generate.stream_generate", return_value=iter(chunks)),
            patch("mlx_vlm.generate._supports_in_place_output", return_value=True),
            contextlib.redirect_stdout(buffer),
        ):
            result = generate(
                Model(),
                FakeProcessor(),
                "",
                verbose=True,
                diffusion_show_unmasking=True,
                diffusion_unmasking_width=20,
            )

        self.assertEqual(result.text, long_text)
        self.assertIn(long_text, buffer.getvalue())

    def test_auto_processor_uses_local_text_only_processor(self):
        from transformers import AutoProcessor

        from mlx_vlm.models.diffusion_gemma4 import DiffusionGemma4Processor

        for model_type in ("diffusion_gemma4", "diffusion_gemma"):
            sentinel = object()
            with TemporaryDirectory() as tmpdir:
                Path(tmpdir, "config.json").write_text(
                    json.dumps({"model_type": model_type})
                )
                with patch.object(
                    DiffusionGemma4Processor,
                    "from_pretrained",
                    return_value=sentinel,
                ) as from_pretrained:
                    processor = AutoProcessor.from_pretrained(tmpdir)

            self.assertIs(processor, sentinel)
            from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
