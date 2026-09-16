"""DiffusionGemma numerical, vision, sanitization, and quantization contracts."""

import unittest

import mlx.core as mx

from mlx_vlm.tests.diffusion_fixtures import make_diffusion_model, tiny_config_dict


class TestDiffusionGemma4(unittest.TestCase):
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

        self.assertLess(float(max_diff.item()), 1e-5)

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
            self.skipTest(
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
        self.assertLess(
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            ),
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
            float(
                np.max(np.abs(hf_logits.detach().cpu().numpy() - np.array(mlx_logits)))
            ),
            1e-5,
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

        self.assertLess(
            float(
                np.max(
                    np.abs(
                        hf_processed.detach().cpu().numpy() - np.array(mlx_processed)
                    )
                )
            ),
            1e-5,
        )
        self.assertEqual(hf_argmax.detach().cpu().numpy().tolist(), mlx_argmax.tolist())
        self.assertEqual(
            hf_current.detach().cpu().numpy().tolist(), mlx_argmax.tolist()
        )

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

        self.assertIn("model.decoder.layers.0.experts.gate_up_proj.weight", sanitized)
        self.assertIn("model.decoder.layers.0.experts.down_proj.weight", sanitized)
        self.assertEqual(
            sanitized["model.decoder.layers.0.experts.gate_up_proj.weight"].shape,
            (4, 16, 16),
        )
        self.assertIn("model.encoder.language_model.layers.0.layer_scalar", sanitized)
        self.assertNotIn(
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight", sanitized
        )
        self.assertNotIn(
            "model.encoder.embed_vision.embedding_projection.weight", sanitized
        )
        self.assertNotIn(
            "model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight",
            sanitized,
        )
        self.assertNotIn("lm_head.weight", sanitized)

    def test_quant_predicate_uses_8bit_for_embeddings_and_attention(self):
        model = make_diffusion_model()
        predicate = model.quant_predicate
        decoder = model.model.decoder

        self.assertEqual(
            predicate("model.decoder.embed_tokens", decoder.embed_tokens),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.self_attn.q_proj",
                decoder.layers[0].self_attn.q_proj,
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.router.proj", decoder.layers[0].router.proj
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertEqual(
            predicate(
                "model.decoder.layers.0.mlp.gate_proj", decoder.layers[0].mlp.gate_proj
            ),
            {"group_size": 64, "bits": 8},
        )
        self.assertIs(
            predicate(
                "model.decoder.layers.0.experts.gate_up_proj",
                decoder.layers[0].experts.gate_up_proj,
            ),
            True,
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
            self.assertEqual(mask.shape, (1, 1, 4, 4))
            # Image tokens attend bidirectionally within the block.
            self.assertTrue(bool(mask[0, 0, 1, 2].item()))
            # Text tokens stay causal.
            self.assertFalse(bool(mask[0, 0, 0, 1].item()))
            self.assertFalse(bool(mask[0, 0, 0, 3].item()))
            # Later text token sees the whole prefix causally.
            self.assertTrue(bool(mask[0, 0, 3, 0].item()))

        # Without vision tokens the fast path is preserved.
        text_masks = encoder._make_encoder_masks(
            h, cache, None, mm_token_type_ids=mx.zeros((1, 4), dtype=mx.int32)
        )
        for mask in text_masks:
            self.assertFalse(isinstance(mask, mx.array) and mask.shape == (1, 1, 4, 4))

    def test_video_features_scattered_into_embeddings(self):
        model = make_diffusion_model(vision=True)
        config = model.config

        input_ids = mx.array([[2, config.video_token_id, 3]])
        pixel_values = mx.random.uniform(shape=(1, 3, 4, 4))

        text_only = model.get_input_embeddings(input_ids=input_ids).inputs_embeds
        with_video = model.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values
        ).inputs_embeds

        self.assertEqual(with_video.shape, text_only.shape)
        self.assertTrue(bool(mx.allclose(with_video[0, 0], text_only[0, 0]).item()))
        self.assertTrue(bool(mx.allclose(with_video[0, 2], text_only[0, 2]).item()))
        self.assertFalse(bool(mx.allclose(with_video[0, 1], text_only[0, 1]).item()))

        expected = model.model.encoder.get_image_features(pixel_values).astype(
            with_video.dtype
        )
        self.assertTrue(
            bool(mx.allclose(with_video[0, 1], expected[0, 0], atol=1e-5).item())
        )

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
        self.assertIn(
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
            sanitized,
        )
        self.assertIn(
            "model.encoder.embed_vision.embedding_projection.weight", sanitized
        )
        # Clipping calibration tensors are dropped when clipped linears are off.
        self.assertNotIn(
            "model.encoder.vision_tower.encoder.layers.0.mlp.gate_proj.input_max",
            sanitized,
        )

        text_model = make_diffusion_model()
        sanitized = text_model.sanitize(dict(weights))
        self.assertEqual(sanitized, {})


if __name__ == "__main__":
    unittest.main()
