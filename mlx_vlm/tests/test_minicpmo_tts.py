import sys
import types
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np


class TestMiniCPMOTTS(unittest.TestCase):
    def test_reference_voice_prompt_preserves_messages_and_audio_order(self):
        from mlx_vlm.models.minicpmo.processing_minicpmo import MiniCPMOProcessor

        processor = MiniCPMOProcessor.__new__(MiniCPMOProcessor)
        user = (
            "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<|tts_bos|>"
        )
        custom_system = "<|im_start|>system\nBe concise.<|im_end|>\n"
        for system in ("", custom_system):
            for input_audio in (None, "input.wav", ["input.wav"]):
                with self.subTest(system=system, audio=input_audio):
                    prompt, paths = processor.prepare_audio_generation(
                        system + user, audio=input_audio, ref_audio_path="voice.wav"
                    )
                    self.assertTrue(
                        prompt.startswith("<|im_start|>system\nClone the voice")
                    )
                    self.assertTrue(prompt.endswith("<|tts_bos|>"))
                    self.assertEqual(prompt.count("<|im_start|>system"), 1)
                    self.assertEqual(processor._count_audio_markers(prompt), len(paths))
                    if system:
                        self.assertIn("Be concise.<|im_end|>", prompt)
                    if input_audio:
                        self.assertEqual(paths, ["voice.wav", "input.wav"])
                        self.assertIn("<|im_start|>user\n<audio>Say hello.", prompt)
                    else:
                        self.assertEqual(paths, ["voice.wav"])
                        self.assertTrue(prompt.endswith(user))
                    if isinstance(input_audio, list):
                        self.assertEqual(input_audio, ["input.wav"])
                    # A caller-supplied system reference must not be duplicated.
                    again = processor.prepare_audio_generation(
                        prompt, audio=paths, ref_audio_path="./voice.wav"
                    )
                    self.assertEqual(again, (prompt, paths))

    def test_audio_input_can_also_supply_reference_voice(self):
        from mlx_vlm.models.minicpmo.processing_minicpmo import MiniCPMOProcessor

        processor = MiniCPMOProcessor.__new__(MiniCPMOProcessor)
        prompt, paths = processor.prepare_audio_generation(
            "<|im_start|>user\n<audio>./</audio>Respond.<|im_end|>\n",
            audio=["input.wav"],
        )
        self.assertEqual(paths, ["input.wav", "input.wav"])
        self.assertEqual(processor._count_audio_markers(prompt), 2)
        self.assertIn("<|im_start|>user\n<audio>Respond.", prompt)

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
            min_new_token=2,
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

            def decode(self, tokens, **kwargs):
                return "speech"

            def convert_tokens_to_ids(self, token):
                return {
                    "<|tts_bos|>": self.tts_start_id,
                    "<|tts_eos|>": self.tts_end_id,
                }.get(token, -1)

        model = Model.__new__(Model)
        from mlx_vlm.models.minicpmo.config import MiniCPMTTSConfig

        model.tts = types.SimpleNamespace(config=MiniCPMTTSConfig())
        model.validate_audio_generation = lambda **kwargs: None
        object.__setattr__(
            model,
            "_speech_vocoder",
            types.SimpleNamespace(
                sample_rate=24000, decode=lambda *args, **kwargs: mx.zeros(24)
            ),
        )
        captured = {}

        def generate_speech_tokens(full_input_ids, **kwargs):
            captured["full_input_ids"] = full_input_ids
            captured.update(kwargs)
            return mx.zeros((1, 1, 1), dtype=mx.int32)

        model.generate_speech_tokens = generate_speech_tokens
        model.find_tts_bound = lambda *args: (2, 3)
        output = Model.generate_audio(
            model,
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
            generated_tokens=[3],
            mask=mx.ones((1, 2), dtype=mx.int32),
            tokenizer=Tokenizer(),
            tts_max_tokens=5,
            tts_temperature=0.2,
            tts_top_p=0.3,
            tts_top_k=6,
            tts_repetition_penalty=1.2,
            max_tokens=99,
        )

        mx.eval(output.audio_tokens)
        self.assertEqual(output.audio_tokens.shape, (1, 1, 1))
        self.assertEqual(captured["tts_max_new_token"], 5)
        self.assertEqual(captured["tts_bound"], (2, 3))
        self.assertEqual(captured["mask"].shape, (1, 3))
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


def _tiny_speech_model():
    from mlx_vlm.models.minicpmo import (
        MiniCPMTTSConfig,
        Model,
        ModelConfig,
        TextConfig,
        VisionConfig,
    )

    return Model(
        ModelConfig(
            text_config=TextConfig(
                model_type="qwen3",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=32,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=64,
                rope_scaling={"type": "default", "mrope_section": [1, 1, 2]},
            ),
            vision_config=VisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
            ),
            tts_config=MiniCPMTTSConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_text_tokens=32,
                num_audio_tokens=16,
                llm_dim=16,
                backbone_vocab_size=32,
                audio_bos_token_id=28,
                text_eos_token_id=29,
                normalize_projected_hidden=True,
            ),
            init_audio=False,
            query_num=4,
        )
    )


def test_speech_hidden_states_match_cached_forward():
    from mlx_lm.models.cache import KVCache

    model = _tiny_speech_model()
    ids = mx.array([[1, 2, 3, 4]])
    full = model.get_hidden_states(ids)
    cache = [KVCache() for _ in model.layers]
    parts = []
    for i in range(ids.shape[1]):
        positions = mx.full((3, 1, 1), i, dtype=mx.int32)
        parts.append(
            model.language_model.model(
                ids[:, i : i + 1], cache=cache, position_ids=positions
            )
        )
    np.testing.assert_allclose(
        np.array(full), np.array(mx.concatenate(parts, axis=1)), atol=1e-5
    )


def test_speech_tokens_run_with_tiny_model():
    from mlx_vlm.models.minicpmo import TTSSamplingParams

    model = _tiny_speech_model()
    ids = mx.array([[1, 10, 3, 4, 11]])
    tokens = model.generate_speech_tokens(
        ids,
        tts_start_id=10,
        tts_end_id=11,
        tts_max_new_token=2,
        tts_sampling_params=TTSSamplingParams(temperature=0),
    )
    assert tokens.shape == (1, 2, 1)
    assert mx.all(tokens < 15).item()


def test_tts_eos_is_never_sent_to_codec():
    from mlx_vlm.models.minicpmo import TTSSamplingParams

    tts = _tiny_speech_model().tts
    with patch.object(tts, "_sample", side_effect=[mx.array([4]), mx.array([15])]):
        result = tts.generate(
            mx.zeros((1, 2, 16)),
            min_new_token=0,
            max_new_token=4,
            sampling_params=TTSSamplingParams(temperature=0),
        )
    assert result.finished
    assert result.new_ids.tolist() == [[[4]]]


def test_tts_bound_rejects_incomplete_latest_turn():
    import pytest

    model = _tiny_speech_model()
    with pytest.raises(ValueError, match="incomplete"):
        model.find_tts_bound(mx.array([[10, 3, 11, 10, 4]]), 10, 11)
    with pytest.raises(ValueError, match="empty"):
        model.find_tts_bound(mx.array([[10, 11]]), 10, 11)


def test_sanitizer_preserves_converted_weights_and_disables_missing_tts(tmp_path):
    import json
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from mlx_vlm.utils import load_model

    model = _tiny_speech_model()
    config = asdict(model.config)
    config.pop("audio_config")
    weights = dict(tree_flatten(model.parameters()))
    sanitized = model.sanitize(weights)
    assert set(sanitized) == set(weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_model(tmp_path)
    assert loaded.supports_audio_generation
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_allclose(
        np.array(loaded.get_hidden_states(ids)),
        np.array(model.get_hidden_states(ids)),
        atol=1e-6,
    )

    # Old 4-bit artifacts still carry init_tts=True, but lack all TTS tensors.
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {k: v for k, v in weights.items() if not k.startswith("tts.")},
    )
    loaded = load_model(tmp_path)
    assert not loaded.supports_audio_generation
    assert loaded.config.init_tts is False
    assert loaded.get_hidden_states(ids).shape == (1, 3, 16)


def test_partial_tts_checkpoint_fails_strict_loading(tmp_path):
    import json
    from dataclasses import asdict

    import pytest
    from mlx.utils import tree_flatten

    from mlx_vlm.utils import load_model

    model = _tiny_speech_model()
    config = asdict(model.config)
    config.pop("audio_config")
    weights = dict(tree_flatten(model.parameters()))
    del weights["tts.head_code.0.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Missing"):
        load_model(tmp_path)


def test_vocoder_keeps_tokens_on_device_and_refreshes_reference(tmp_path):
    from unittest.mock import Mock

    from mlx_vlm.models.minicpmo.vocoder import StepAudio2Vocoder

    voice_a, voice_b = tmp_path / "a.wav", tmp_path / "b.wav"
    voice_a.touch()
    voice_b.touch()
    vocoder = StepAudio2Vocoder.__new__(StepAudio2Vocoder)
    vocoder.n_timesteps = 10
    vocoder._codec = Mock(return_value=mx.zeros((1, 24)))
    tokens = mx.array([[[1], [2]]])
    for voice in (voice_a, voice_b):
        wav = vocoder.decode(tokens, prompt_wav_path=str(voice))
        assert wav.shape == (24,)
        assert vocoder._codec.call_args.kwargs["use_cache"] is False
        assert isinstance(vocoder._codec.call_args.args[0], mx.array)
        assert vocoder._codec.call_args.args[0].tolist() == [1, 2]
        assert vocoder._codec.call_args.args[1] == str(voice)


def test_missing_reference_fails_before_text_generation(monkeypatch):
    from unittest.mock import Mock

    import pytest

    from mlx_vlm import generate_audio
    from mlx_vlm.generate import dispatch

    text = Mock()
    monkeypatch.setattr(dispatch, "generate", text)
    with pytest.raises(ValueError, match="ref_audio_path"):
        generate_audio(_tiny_speech_model(), None, "prompt")
    text.assert_not_called()


def test_old_tts_turn_cannot_supply_new_response_audio(tmp_path):
    from unittest.mock import Mock

    import pytest

    model = _tiny_speech_model()
    ref = tmp_path / "voice.wav"
    ref.touch()
    tokenizer = types.SimpleNamespace(tts_start_id=10, tts_end_id=11)
    with patch.object(model, "generate_speech_tokens", Mock()) as generate_tokens:
        with pytest.raises(ValueError, match="new response"):
            model.generate_audio(
                input_ids=mx.array([[10, 3, 11, 2]]),
                generated_tokens=[4],
                tokenizer=tokenizer,
                ref_audio_path=str(ref),
            )
    generate_tokens.assert_not_called()
