import unittest

import mlx.core as mx

from mlx_vlm.models.qwen3_omni_moe.config import (
    AudioConfig,
    Code2WavConfig,
    CodePredictorConfig,
    ModelConfig,
    TalkerConfig,
    TextConfig,
    ThinkerConfig,
    VisionConfig,
)
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model

IMAGE_TOKEN = 60
VISION_START = 63
VISION_END = 59


def _tiny_text_config(model_type="qwen3_omni_moe_text_encoder"):
    return TextConfig(
        model_type=model_type,
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=0,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=32,
        rms_norm_eps=1e-5,
        vocab_size=64,
        rope_theta=10000,
        max_position_embeddings=64,
    )


def _tiny_model(vision_config=None, **thinker_kwargs):
    text_config = _tiny_text_config()
    thinker_config = ThinkerConfig(
        text_config=text_config,
        vision_config=vision_config
        or VisionConfig(
            depth=0,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
            image_size=8,
            patch_size=2,
            spatial_patch_size=2,
            spatial_merge_size=1,
            in_channels=3,
            in_chans=3,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
        audio_config=AudioConfig(
            d_model=16,
            encoder_layers=0,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            num_hidden_layers=0,
            num_mel_bins=8,
            output_dim=16,
            downsample_hidden_size=8,
        ),
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
        **thinker_kwargs,
    )
    talker_config = TalkerConfig(
        text_config=_tiny_text_config("qwen3_omni_moe_talker_text"),
        code_predictor_config=CodePredictorConfig(
            num_hidden_layers=1,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            num_code_groups=2,
        ),
        accept_hidden_layer=0,
        thinker_hidden_size=16,
    )
    code2wav_config = Code2WavConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_dim=16,
        codebook_dim=8,
        codebook_size=32,
        num_quantizers=2,
        num_semantic_quantizers=1,
        semantic_codebook_size=32,
        vector_quantization_hidden_dimension=8,
    )
    return Model(
        ModelConfig(
            thinker_config=thinker_config,
            talker_config=talker_config,
            code2wav_config=code2wav_config,
            enable_audio_output=False,
            im_start_token_id=10,
            im_end_token_id=11,
            system_token_id=12,
            user_token_id=13,
            assistant_token_id=14,
            tts_bos_token_id=15,
            tts_eos_token_id=16,
            tts_pad_token_id=17,
        )
    )


def _tiny_vision_model():
    return _tiny_model(
        vision_config=VisionConfig(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
            image_size=8,
            patch_size=2,
            spatial_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            in_chans=3,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )


def _image_inputs():
    # grid 1x4x4 patches, merge 2 -> 4 visual tokens
    mx.random.seed(7)
    pixel_values = mx.random.normal((16, 24))
    input_ids = mx.array(
        [[1, 2, VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END, 3, 4, 5]],
        dtype=mx.int32,
    )
    return input_ids, pixel_values, mx.array([[1, 4, 4]])


class Qwen3OmniMoeTest(unittest.TestCase):
    def test_thinker_generation_keeps_hidden_states_aligned(self):
        model = _tiny_model()
        input_ids = mx.array([[10, 13, 20, 11, 10, 14]], dtype=mx.int32)

        sequences, hidden_states, input_embeds = (
            model._generate_thinker_with_hidden_states(
                input_ids,
                target_layer_idx=0,
                thinker_max_new_tokens=3,
                thinker_eos_token_id=-1,
            )
        )
        expected_hidden_states, expected_input_embeds = (
            model.extract_thinker_hidden_states(
                sequences,
                target_layer_idx=0,
            )
        )
        mx.eval(sequences, hidden_states, input_embeds)
        mx.eval(expected_hidden_states, expected_input_embeds)

        self.assertEqual(sequences.shape[1], input_ids.shape[1] + 3)
        self.assertEqual(hidden_states.shape, expected_hidden_states.shape)
        self.assertEqual(input_embeds.shape, expected_input_embeds.shape)
        self.assertTrue(
            bool(
                mx.allclose(
                    hidden_states,
                    expected_hidden_states,
                    rtol=1e-4,
                    atol=1e-4,
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.allclose(
                    input_embeds,
                    expected_input_embeds,
                    rtol=1e-6,
                    atol=1e-6,
                ).item()
            )
        )

    def test_deepstack_embeds_reach_language_model(self):
        model = _tiny_vision_model()
        input_ids, pixel_values, grid = _image_inputs()

        features = model.thinker.get_input_embeddings(
            input_ids, pixel_values=pixel_values, image_grid_thw=grid
        )
        embeds = features.deepstack_visual_embeds
        self.assertIsNotNone(embeds)
        self.assertEqual(len(embeds), 2)
        for e in embeds:
            self.assertEqual(tuple(e.shape), (4, 16))

        with_injection = model(input_ids, pixel_values, image_grid_thw=grid).logits
        model_cls = type(model.thinker.language_model.model)
        orig = model_cls._deepstack_process
        model_cls._deepstack_process = (
            lambda self, hidden_states, *a, **k: hidden_states
        )
        try:
            without_injection = model(
                input_ids, pixel_values, image_grid_thw=grid
            ).logits
        finally:
            model_cls._deepstack_process = orig
        mx.eval(with_injection, without_injection)
        self.assertFalse(
            bool(mx.allclose(with_injection, without_injection, atol=1e-6).item())
        )

    def test_deepstack_injection_is_batch_safe(self):
        model = _tiny_vision_model()
        input_ids, pixel_values, grid = _image_inputs()
        text_ids = mx.array([[1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]], dtype=mx.int32)

        solo_image = model(input_ids, pixel_values, image_grid_thw=grid).logits
        solo_text = model(text_ids).logits
        batch = model(
            mx.concatenate([input_ids, text_ids], axis=0),
            pixel_values,
            image_grid_thw=grid,
        ).logits
        mx.eval(solo_image, solo_text, batch)

        self.assertTrue(
            bool(mx.allclose(batch[0:1], solo_image, rtol=1e-4, atol=1e-5).item())
        )
        self.assertTrue(
            bool(mx.allclose(batch[1:2], solo_text, rtol=1e-4, atol=1e-5).item())
        )

    def test_quant_predicate_forwarded_to_top_level_model(self):
        model = _tiny_model()
        predicate = model.quant_predicate
        self.assertIsNotNone(predicate)
        self.assertEqual(
            predicate("thinker.language_model.model.layers.0.mlp.gate", None),
            {"group_size": 64, "bits": 8},
        )
        self.assertTrue(
            predicate("thinker.language_model.model.layers.0.self_attn.q_proj", None)
        )


if __name__ == "__main__":
    unittest.main()
