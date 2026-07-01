import sys
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np


class TestMiniCPMOTTS(unittest.TestCase):
    def test_tts_config_parses(self):
        from mlx_vlm.models.minicpmo.config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "model_type": "minicpmo",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "head_dim": 4,
                "rope_theta": 10000.0,
                "max_position_embeddings": 64,
                "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                },
                "tts_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_text_tokens": 64,
                    "num_audio_tokens": 32,
                    "llm_dim": 8,
                    "condition_type": "hidden_text_merge",
                    "normalize_projected_hidden": True,
                },
            }
        )

        self.assertEqual(cfg.tts_config.hidden_size, 16)
        self.assertEqual(cfg.tts_config.num_audio_tokens, 32)
        self.assertEqual(cfg.tts_config.condition_type, "hidden_text_merge")
        self.assertTrue(cfg.tts_config.normalize_projected_hidden)

    def test_tiny_tts_generates_audio_tokens(self):
        from mlx_vlm.models.minicpmo.config import MiniCPMTTSConfig
        from mlx_vlm.models.minicpmo.tts import MiniCPMTTS, TTSSamplingParams

        cfg = MiniCPMTTSConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=1,
            num_text_tokens=32,
            num_audio_tokens=16,
            llm_dim=8,
        )
        model = MiniCPMTTS(cfg)
        out = model.generate(
            mx.zeros((1, 3, 16)),
            max_new_token=2,
            min_new_token=0,
            sampling_params=TTSSamplingParams(
                temperature=0.0,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
            ),
        )
        mx.eval(out.new_ids)
        self.assertEqual(out.new_ids.shape, (1, 2, 1))

    def test_sanitize_materializes_tts_weight_norm(self):
        from mlx_vlm.models.minicpmo.config import (
            MiniCPMTTSConfig,
            ModelConfig,
            TextConfig,
            VisionConfig,
        )
        from mlx_vlm.models.minicpmo.minicpmo import Model

        text = TextConfig(
            model_type="qwen3",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=20,
            num_key_value_heads=2,
            head_dim=4,
            rope_theta=10000,
            max_position_embeddings=64,
        )
        vision = VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        tts = MiniCPMTTSConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            num_text_tokens=20,
            num_audio_tokens=12,
            llm_dim=8,
        )
        model = Model(
            ModelConfig(
                text_config=text,
                vision_config=vision,
                tts_config=tts,
                init_audio=False,
                init_tts=True,
            )
        )
        weights = {
            "tts.head_code.0.parametrizations.weight.original0": mx.ones((12, 1)),
            "tts.head_code.0.parametrizations.weight.original1": mx.ones((12, 8)),
        }
        sanitized = model.sanitize(weights)

        self.assertIn("tts.head_code.0.weight", sanitized)
        self.assertEqual(sanitized["tts.head_code.0.weight"].shape, (12, 8))

    def test_processor_exposes_tts_and_spk_tokens(self):
        from mlx_vlm.models.minicpmo.processing_minicpmo import MiniCPMOProcessor

        class Tokenizer:
            eos_token = "<eos>"
            pad_token = None
            pad_token_id = 0

            def __init__(self):
                self.vocab = {
                    "<|spk_bos|>": 10,
                    "<|spk_eos|>": 11,
                    "<|tts_bos|>": 12,
                    "<|tts_eos|>": 13,
                    "<|listen|>": 14,
                    "<image>": 20,
                    "</image>": 21,
                    "<slice>": 22,
                    "</slice>": 23,
                    "<|audio_start|>": 30,
                    "<|audio_end|>": 31,
                }

            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, -1)

        processor = MiniCPMOProcessor.__new__(MiniCPMOProcessor)
        processor.tokenizer = Tokenizer()
        processor._ensure_tokenizer_attrs()

        self.assertEqual(processor.tokenizer.tts_start_id, 12)
        self.assertEqual(processor.tokenizer.tts_end_id, 13)

        ids = np.array([1, 10, 2, 3, 11, 4], dtype=np.int32)
        np.testing.assert_array_equal(
            processor._compute_spk_bounds(ids),
            np.array([[2, 4]], dtype=np.int32),
        )

    def test_model_generate_audio_consumes_tts_kwargs(self):
        from mlx_vlm.models.minicpmo.minicpmo import Model

        class Tokenizer:
            tts_start_id = 10
            tts_end_id = 11

            def convert_tokens_to_ids(self, token):
                return {
                    "<|tts_bos|>": self.tts_start_id,
                    "<|tts_eos|>": self.tts_end_id,
                }.get(token, -1)

        model = Model.__new__(Model)
        captured = {}

        def generate_speech_tokens(full_input_ids, **kwargs):
            captured["full_input_ids"] = full_input_ids
            captured.update(kwargs)
            return mx.zeros((1, 1, 1), dtype=mx.int32)

        model.generate_speech_tokens = generate_speech_tokens
        output = Model.generate_audio(
            model,
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
            generated_tokens=[3],
            tokenizer=Tokenizer(),
            tts_min_tokens=4,
            tts_max_tokens=5,
            tts_temperature=0.2,
            tts_top_p=0.3,
            tts_top_k=6,
            tts_repetition_penalty=1.2,
            max_tokens=99,
        )

        mx.eval(output.audio_tokens)
        self.assertEqual(output.audio_tokens.shape, (1, 1, 1))
        self.assertEqual(captured["tts_min_new_token"], 4)
        self.assertEqual(captured["tts_max_new_token"], 5)
        self.assertEqual(captured["tts_start_id"], 10)
        self.assertEqual(captured["tts_end_id"], 11)
        self.assertEqual(captured["max_tokens"], 99)
        params = captured["tts_sampling_params"]
        self.assertEqual(params.temperature, 0.2)
        self.assertEqual(params.top_p, 0.3)
        self.assertEqual(params.top_k, 6)
        self.assertEqual(params.repetition_penalty, 1.2)

    def test_stepaudio2_vocoder_uses_codec_default_repo(self):
        from mlx_vlm.models.minicpmo.vocoder import StepAudio2Vocoder

        calls = []

        class Codec:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls.append((args, kwargs))
                return cls()

        stepaudio2 = types.ModuleType("mlx_audio.codec.models.stepaudio2")
        stepaudio2.StepAudio2Token2Wav = Codec

        with patch.dict(
            sys.modules,
            {
                "mlx_audio": types.ModuleType("mlx_audio"),
                "mlx_audio.codec": types.ModuleType("mlx_audio.codec"),
                "mlx_audio.codec.models": types.ModuleType("mlx_audio.codec.models"),
                "mlx_audio.codec.models.stepaudio2": stepaudio2,
            },
        ):
            StepAudio2Vocoder()

        self.assertEqual(calls, [((), {})])


if __name__ == "__main__":
    unittest.main()
