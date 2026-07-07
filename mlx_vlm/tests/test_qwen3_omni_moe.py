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


def _tiny_model():
    text_config = _tiny_text_config()
    thinker_config = ThinkerConfig(
        text_config=text_config,
        vision_config=VisionConfig(
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


class Qwen3OmniMoeTest(unittest.TestCase):
    def test_thinker_generation_keeps_hidden_states_aligned(self):
        model = _tiny_model()
        input_ids = mx.array([[10, 13, 20, 11, 10, 14]], dtype=mx.int32)

        sequences, hidden_states, input_embeds = (
            model._generate_thinker_with_hidden_states(
                input_ids,
                target_layer_idx=0,
                thinker_max_new_tokens=3,
                thinker_eos_token_id=63,
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


if __name__ == "__main__":
    unittest.main()
