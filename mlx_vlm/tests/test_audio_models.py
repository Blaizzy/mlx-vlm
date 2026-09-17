"""Audio and omni model components, speech generation, and voice sessions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import load
from mlx_vlm.models.gemma3.config import TextConfig as Gemma3TextConfig
from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.nemotron_h.language import NemotronHModel
from mlx_vlm.models.nemotron_h_nano_omni.audio import (
    SoundFeatureExtractor,
    sanitize_audio_weights,
)
from mlx_vlm.models.nemotron_h_nano_omni.config import (
    AudioConfig as NemotronAudioConfig,
)
from mlx_vlm.models.nemotron_h_nano_omni.config import (
    ModelConfig as NemotronModelConfig,
)
from mlx_vlm.models.nemotron_h_nano_omni.config import TextConfig as NemotronTextConfig
from mlx_vlm.models.nemotron_h_nano_omni.config import (
    VisionConfig as NemotronVisionConfig,
)
from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import (
    Model as NemotronModel,
)
from mlx_vlm.models.nemotron_voicechat import convert
from mlx_vlm.models.nemotron_voicechat.config import CharacterEncoderConfig, MoGConfig
from mlx_vlm.models.nemotron_voicechat.convert import build_runtime_config
from mlx_vlm.models.nemotron_voicechat.model import Model as VoiceChatModel
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession
from mlx_vlm.models.nemotron_voicechat.streaming import (
    VoiceChatFrameTiming,
    VoiceChatProfile,
)
from mlx_vlm.models.nemotron_voicechat.tts import CharAwareSubwordEncoder, MoGHead
from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig as QwenAudioConfig
from mlx_vlm.models.qwen3_omni_moe.config import Code2WavConfig as QwenCode2WavConfig
from mlx_vlm.models.qwen3_omni_moe.config import (
    CodePredictorConfig as QwenCodePredictorConfig,
)
from mlx_vlm.models.qwen3_omni_moe.config import ModelConfig as QwenModelConfig
from mlx_vlm.models.qwen3_omni_moe.config import TalkerConfig as QwenTalkerConfig
from mlx_vlm.models.qwen3_omni_moe.config import TextConfig as QwenTextConfig
from mlx_vlm.models.qwen3_omni_moe.config import ThinkerConfig as QwenThinkerConfig
from mlx_vlm.models.qwen3_omni_moe.config import VisionConfig as QwenVisionConfig
from mlx_vlm.models.qwen3_omni_moe.qwen3_omni_moe import Model as QwenModel


def _small_config(factory, **overrides):
    return factory(
        **(
            dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
            )
            | overrides
        )
    )


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

    def test_tiny_tts_generates_audio_tokens(self):
        from mlx_vlm.models.minicpmo.config import MiniCPMTTSConfig
        from mlx_vlm.models.minicpmo.tts import MiniCPMTTS, TTSSamplingParams

        cfg = _small_config(
            MiniCPMTTSConfig,
            num_attention_heads=4,
            num_key_value_heads=4,
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
                temperature=0.0, top_p=None, top_k=None, repetition_penalty=None
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
            processor._compute_spk_bounds(ids), np.array([[2, 4]], dtype=np.int32)
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
            text_config=_small_config(
                TextConfig,
                model_type="qwen3",
                num_hidden_layers=2,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                vocab_size=32,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=64,
                rope_scaling={"type": "default", "mrope_section": [1, 1, 2]},
            ),
            vision_config=_small_config(VisionConfig, image_size=28),
            tts_config=_small_config(
                MiniCPMTTSConfig,
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
    from mlx_vlm.models.cache import KVCache

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


def test_speech_generation_without_mlx_lm():
    # Use a fresh interpreter so an installed or already-imported mlx-lm cannot
    # hide an undeclared dependency, as it did in the original local test run.
    script = """
import sys
sys.modules["mlx_lm"] = None

import mlx.core as mx
from mlx_vlm.models.minicpmo import TTSSamplingParams
from mlx_vlm.tests.test_audio_models import _tiny_speech_model

model = _tiny_speech_model()
tokens = model.generate_speech_tokens(
    mx.array([[1, 10, 3, 4, 11]]),
    tts_start_id=10,
    tts_end_id=11,
    tts_max_new_token=2,
    tts_sampling_params=TTSSamplingParams(temperature=0.8, top_p=0.85, top_k=5),
)
mx.eval(tokens)
assert tokens.shape == (1, 2, 1)
assert mx.all(tokens < 15).item()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


IMAGE_TOKEN = 60
VISION_START = 63
VISION_END = 59


def _tiny_text_config(model_type="qwen3_omni_moe_text_encoder"):
    return _small_config(
        QwenTextConfig,
        model_type=model_type,
        num_hidden_layers=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=0,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=32,
        rms_norm_eps=1e-05,
        vocab_size=64,
        rope_theta=10000,
        max_position_embeddings=64,
    )


def _qwen_vision_config(**overrides):
    return QwenVisionConfig(
        **(
            dict(
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
            )
            | overrides
        )
    )


def _tiny_model(vision_config=None, **thinker_kwargs):
    text_config = _tiny_text_config()
    thinker_config = QwenThinkerConfig(
        text_config=text_config,
        vision_config=vision_config or _qwen_vision_config(),
        audio_config=QwenAudioConfig(
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
    talker_config = QwenTalkerConfig(
        text_config=_tiny_text_config("qwen3_omni_moe_talker_text"),
        code_predictor_config=_small_config(
            QwenCodePredictorConfig,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            num_code_groups=2,
        ),
        accept_hidden_layer=0,
        thinker_hidden_size=16,
    )
    code2wav_config = _small_config(
        QwenCode2WavConfig,
        num_key_value_heads=2,
        decoder_dim=16,
        codebook_dim=8,
        codebook_size=32,
        num_quantizers=2,
        num_semantic_quantizers=1,
        semantic_codebook_size=32,
        vector_quantization_hidden_dimension=8,
    )
    return QwenModel(
        QwenModelConfig(
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
        vision_config=_qwen_vision_config(
            depth=2, spatial_merge_size=2, deepstack_visual_indexes=[0, 1]
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
            model.extract_thinker_hidden_states(sequences, target_layer_idx=0)
        )
        mx.eval(sequences, hidden_states, input_embeds)
        mx.eval(expected_hidden_states, expected_input_embeds)

        self.assertEqual(sequences.shape[1], input_ids.shape[1] + 3)
        self.assertEqual(hidden_states.shape, expected_hidden_states.shape)
        self.assertEqual(input_embeds.shape, expected_input_embeds.shape)
        self.assertTrue(
            bool(
                mx.allclose(
                    hidden_states, expected_hidden_states, rtol=1e-4, atol=1e-4
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.allclose(
                    input_embeds, expected_input_embeds, rtol=1e-6, atol=1e-6
                ).item()
            )
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

    def _thinker_logits(self, model, ids, cache, **kw):
        x = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
        out = model.thinker(x, cache=cache, **kw)
        return out.logits if hasattr(out, "logits") else out

    def _assert_prefill_decode_match(self, pre, step, pos):
        a, b = pre.reshape(-1), step[:, -1].reshape(-1)
        cosine = float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-9))
        self.assertGreater(cosine, 0.999, f"decode diverges from prefill at pos {pos}")

    def test_decode_continues_rope_positions_from_cache_offset(self):
        # Prefill and step-by-step decode of the same text tokens must agree.
        # get_rope_index restarts positions at 0 on every call, so without
        # cache-offset-aware positions each decoded token collapses to RoPE
        # position 0 and diverges from prefill (#1983).
        from mlx_vlm.models.cache import make_prompt_cache

        model = _tiny_model()
        inner = model.thinker.language_model.model
        ids = [20, 21, 22, 23, 24, 25, 26, 27]
        pre = self._thinker_logits(model, [ids], make_prompt_cache(inner))
        cache = make_prompt_cache(inner)
        for pos, tok in enumerate(ids):
            step = self._thinker_logits(model, [[tok]], cache)
            self._assert_prefill_decode_match(pre[:, pos], step, pos)

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


def tiny_text_config(hidden_size=24):
    return NemotronTextConfig(
        model_type="nemotron_h",
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_bias=False,
        mamba_num_heads=1,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=3,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=False,
        hybrid_override_pattern=["*"],
    )


def test_nemotron_h_embeddingless_backbone_requires_inputs_embeds():
    config = tiny_text_config()
    model = NemotronHModel(config, with_embeddings=False)
    inputs_embeds = mx.zeros((1, 2, config.hidden_size))

    output = model(inputs_embeds=inputs_embeds)

    assert output.shape == inputs_embeds.shape
    with pytest.raises(ValueError, match="no token embedding"):
        model(mx.array([[1, 2]], dtype=mx.int32))


def tiny_vision_config():
    return _small_config(
        NemotronVisionConfig,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        max_resolution=32,
        args={"teachers": [{"name": "test"}]},
    )


def tiny_sound_config(hidden_size=16):
    return NemotronAudioConfig(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        projection_hidden_size=32,
    )


def test_sound_feature_extractor_shapes_and_masks():
    pytest.importorskip("mlx_audio")
    config = NemotronAudioConfig()
    extractor = SoundFeatureExtractor(config)
    waveform = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    features, mask, lengths = extractor([waveform])

    assert features.shape == (1, 11, config.num_mel_bins)
    assert mask.shape == (1, 11)
    assert lengths.tolist() == [11]
    assert mask.sum(axis=1).tolist() == [10]
    assert np.isfinite(np.array(features)).all()


def test_audio_path_handles_nvfp4_uint8_lm_head_scales():
    """Regression: nvfp4-quantized models pack lm_head.scales as uint8.

    Before the fix, ``_extract_sound_features`` would cast the audio
    ``input_features`` to ``scales.dtype`` and then crash inside the
    subsampling Conv2d with::

        ValueError: [conv] Invalid input array with type uint8.

    The fix falls back to ``mx.bfloat16`` when ``scales.dtype`` is an
    integer packing type (uint8 for nvfp4, uint32 for some other packed
    modes). This test simulates that quant layout by attaching a uint8
    ``scales`` attribute to a plain ``lm_head`` and verifying the audio
    path completes without dtype error.
    """
    model = NemotronModel(
        NemotronModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=tiny_sound_config(),
            projector_hidden_size=32,
            vit_hidden_size=16,
            img_context_token_id=98,
            sound_context_token_id=99,
        )
    )
    model.eval()

    # Simulate an nvfp4-quantized lm_head: presence of .scales with uint8 dtype.
    model.language_model.lm_head.scales = mx.zeros((1,), dtype=mx.uint8)

    input_ids = mx.array([[1, 99, 99, 99, 2]])
    input_features = mx.random.normal((1, 17, 16))
    feature_attention_mask = mx.ones((1, 17), dtype=mx.int32)

    # Pre-fix: this raised ValueError from mx.conv2d.
    output = model.get_input_embeddings(
        input_ids,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )
    assert np.isfinite(np.array(output.inputs_embeds)).all()


def test_model_rejects_sound_token_feature_count_mismatch():
    model = NemotronModel(
        NemotronModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=tiny_sound_config(),
            projector_hidden_size=32,
            vit_hidden_size=16,
            img_context_token_id=98,
            sound_context_token_id=99,
        )
    )
    model.eval()

    with pytest.raises(ValueError, match="Sound token count"):
        model.get_input_embeddings(
            mx.array([[1, 99, 99, 2]]),
            input_features=mx.random.normal((1, 17, 16)),
            feature_attention_mask=mx.ones((1, 17), dtype=mx.int32),
        )


def test_sanitize_audio_and_projection_weights():
    weights = {
        "sound_encoder.encoder.feature_extractor.window": mx.ones((2,)),
        "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked": mx.array(0),
        "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight": mx.zeros(
            (8, 4, 3)
        ),
        "sound_encoder.encoder.subsampling.layers.0.weight": mx.zeros((4, 1, 3, 3)),
        "mlp1.0.weight": mx.ones((4,)),
        "mlp1.1.weight": mx.ones((4, 4)),
        "mlp1.3.weight": mx.ones((4, 4)),
    }

    audio_sanitized = sanitize_audio_weights(weights)
    assert "sound_encoder.encoder.feature_extractor.window" not in audio_sanitized
    assert (
        "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked"
        not in audio_sanitized
    )
    assert audio_sanitized[
        "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight"
    ].shape == (8, 3, 4)
    assert audio_sanitized[
        "sound_encoder.encoder.subsampling.layers.0.weight"
    ].shape == (4, 3, 3, 1)

    model = NemotronModel(
        NemotronModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            sound_config=None,
            projector_hidden_size=4,
            vit_hidden_size=1,
            img_context_token_id=98,
        )
    )
    model_sanitized = model.sanitize(weights)
    assert "mlp1.layers.0.weight" in model_sanitized
    assert "mlp1.layers.1.weight" in model_sanitized
    assert "mlp1.layers.3.weight" in model_sanitized


def test_streaming_profile_summarizes_synchronized_stage_timings():
    profile = VoiceChatProfile(
        frame_duration_ms=80.0,
        frames=[
            VoiceChatFrameTiming(0, 10, 2, 20, 30, 8, 72),
            VoiceChatFrameTiming(1, 8, 1, 16, 24, 6, 56),
            VoiceChatFrameTiming(2, 9, 1, 18, 27, 7, 63),
        ],
    )
    summary = profile.summary(drop_first=1)

    assert summary["frames"] == 2
    assert summary["dropped_cold_frames"] == 1
    assert summary["stages"]["total"]["mean_ms"] == 59.5
    assert summary["processing_frames_per_second"] == pytest.approx(1000 / 59.5)
    assert summary["realtime_factor"] == pytest.approx(59.5 / 80)


def test_character_aware_encoder_prepares_and_scatters_subwords():
    config = CharacterEncoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        char_vocab_size=4,
    )
    encoder = CharAwareSubwordEncoder(config, out_size=8, vocab_size=6)
    encoder.set_vocabulary({"a": 0, "b": 1, "c": 2, "ab": 3, "<s>": 4, "</s>": 5})
    ids = mx.array([[3, 4, 5]], dtype=mx.int32)
    mask = mx.array([[True, False, True]])
    output = encoder(ids, mask)
    mx.eval(output)
    assert output.shape == (1, 3, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_mog_head_inference_shapes_and_finite_values():
    config = MoGConfig(
        intermediate_size=16, low_rank=2, num_layers=1, num_predictions=4
    )
    head = MoGHead(hidden_size=8, out_size=4, config=config)
    inputs = mx.zeros((2, 1, 8))
    mean, logs = head.infer(inputs, guidance_scale=0.2, top_p=0.95)
    mx.eval(mean, logs)
    assert mean.shape == (1, 1, 4)
    assert logs.shape == (1, 1, 1)
    assert bool(mx.all(mx.isfinite(mean)))
    assert bool(mx.all(mx.isfinite(logs)))


def test_gemma3_can_preserve_caller_supplied_embedding_scale():
    config = Gemma3TextConfig(
        model_type="gemma3_text",
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=8,
        sliding_window_pattern=1,
    )

    class Capture:
        def __call__(self, inputs, mask=None, cache=None):
            del mask, cache
            self.inputs = inputs
            return inputs

    class Identity:
        def __call__(self, inputs):
            return inputs

    model = Gemma3Model(config, scale_inputs_embeds=False)
    capture = Capture()
    model.layers = [capture]
    model.norm = Identity()
    inputs = mx.ones((1, 1, config.hidden_size))
    output = model(None, inputs_embeds=inputs)

    assert mx.array_equal(capture.inputs, inputs)
    assert mx.array_equal(output, inputs)


def test_model_creates_session_from_wrapped_tokenizer():
    vocabulary = {"hello": 0}

    class Tokenizer:
        def encode(self, *_args, **_kwargs):
            return [0]

        def decode(self, *_args, **_kwargs):
            return "hello"

        def get_vocab(self):
            return vocabulary

    class TTS:
        def set_vocabulary(self, value):
            self.vocabulary = value

    tts = TTS()
    model = SimpleNamespace(tts_model=SimpleNamespace(tts_model=tts))
    processor = SimpleNamespace(tokenizer=Tokenizer())

    created = VoiceChatModel.create_session(model, processor)

    assert isinstance(created, VoiceChatSession)
    assert created.model is model
    assert created.tokenizer is processor.tokenizer
    assert tts.vocabulary == vocabulary


def test_model_create_session_requires_tokenizer_interface():
    with pytest.raises(TypeError, match="tokenizer-like processor"):
        VoiceChatModel.create_session(SimpleNamespace(), object())


@pytest.mark.skipif(
    not os.environ.get("VOICECHAT_MODEL_PATH"),
    reason="set VOICECHAT_MODEL_PATH to run the real-checkpoint smoke test",
)
def test_real_checkpoint_offline_smoke():
    model_path = Path(os.environ["VOICECHAT_MODEL_PATH"])
    audio_path = os.environ.get("VOICECHAT_AUDIO_PATH")
    if not audio_path:
        pytest.skip("set VOICECHAT_AUDIO_PATH to an input wav")
    model, processor = load(str(model_path), lazy=True)
    result = model.create_session(processor).generate(
        audio_path, system_prompt="Answer briefly.", max_frames=2
    )
    mx.eval(result.audio)
    assert result.audio.shape == (3528,)
    assert result.audio_codes.shape == (2, 31)
    assert result.sample_rate == 22_050
    assert bool(mx.all(mx.isfinite(result.audio)))


@pytest.mark.skipif(
    not os.environ.get("VOICECHAT_MODEL_PATH"),
    reason="set VOICECHAT_MODEL_PATH to run the real-checkpoint smoke test",
)
def test_real_checkpoint_streaming_first_frame_matches_offline():
    from mlx_audio.stt.utils import load_audio

    model_path = Path(os.environ["VOICECHAT_MODEL_PATH"])
    audio_path = os.environ.get("VOICECHAT_AUDIO_PATH")
    if not audio_path:
        pytest.skip("set VOICECHAT_AUDIO_PATH to an input wav")
    model, processor = load(str(model_path), lazy=True)
    session = model.create_session(processor)
    audio = load_audio(audio_path, sr=16_000).squeeze()[:1280]
    offline = session.generate(
        audio, system_prompt="Answer briefly.", max_frames=2, seed=0
    )
    stream = session.create_streaming_session(
        system_prompt="Answer briefly.", seed=0, max_streaming_seconds=1.0
    )
    events = stream.push_audio(audio, sample_rate=16_000)
    audio_event = next(event for event in events if event.kind == "audio")

    assert stream._text_tokens[-1] == int(offline.text_tokens[0])
    assert stream._function_tokens[-1] == int(offline.function_tokens[0])
    assert mx.array_equal(audio_event.audio_codes, offline.audio_codes[0])
    assert mx.allclose(audio_event.samples, offline.audio[:1764], atol=2e-4)


def _source_config():
    return {
        "data": {
            "source_sample_rate": 16_000,
            "target_sample_rate": 22_050,
            "frame_length": 0.08,
        },
        "model": {
            "stt": {
                "model": {
                    "duplex_function_channel_weight": 2.0,
                    "perception": {
                        "output_dim": 4_480,
                        "preprocessor": {
                            "_target_": "Preprocessor",
                            "sample_rate": 16_000,
                            "features": 128,
                            "n_fft": 512,
                        },
                        "encoder": {
                            "_target_": "Conformer",
                            "feat_in": 128,
                            "n_layers": 24,
                            "d_model": 1_024,
                            "n_heads": 8,
                            "att_context_size": [70, 0],
                        },
                    },
                }
            },
            "speech_generation": {
                "data": {"audio_prompt_duration": 3.0, "target_sample_rate": 22_050},
                "model": {
                    "inference_guidance_scale": 0.2,
                    "inference_top_p_or_k": 0.95,
                    "inference_noise_scale": 0.001,
                    "codec_config": {
                        "base_hidden_size": 384,
                        "channel_mult": [1, 2, 4],
                        "rates": [7, 7, 9],
                        "num_blocks": 3,
                        "kernel_size": 7,
                        "latent_size": 512,
                        "n_fft": 16,
                        "hop_length": 4,
                        "num_quantizers": 31,
                        "codebook_size": 1_024,
                    },
                    "tts_config": {
                        "hidden_size": 1_152,
                        "latent_size": 512,
                        "num_quantizers": 31,
                        "codebook_size": 1_024,
                        "num_delay_speech_tokens": 2,
                        "exponent": 3.0,
                        "disable_eos_prediction": True,
                        "use_gated_fusion_for_text_audio": True,
                        "use_subword_flag_emb": True,
                        "use_bos_eos_emb": True,
                        "use_audio_prompt_frozen_projection": True,
                        "backbone_config": {
                            "hidden_size": 1_152,
                            "intermediate_size": 4_608,
                            "num_hidden_layers": 28,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 16,
                            "head_dim": 72,
                            "sliding_window": 7_500,
                        },
                        "cas_config": {
                            "backbone_config": {
                                "encoder": {
                                    "hidden_size": 1_152,
                                    "intermediate_size": 4_608,
                                    "num_hidden_layers": 1,
                                    "num_attention_heads": 16,
                                    "num_key_value_heads": 16,
                                    "head_dim": 72,
                                }
                            }
                        },
                        "mog_head_config": {
                            "intermediate_size": 4_608,
                            "low_rank": 64,
                            "min_log_std": -4.0,
                            "num_layers": 3,
                            "num_predictions": 1_024,
                        },
                    },
                },
            },
        },
        "_rnnt_merge_info": {
            "decoder_config": {
                "vocab_size": 1_024,
                "blank_as_pad": True,
                "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
            },
            "joint_config": {
                "num_classes": 1_024,
                "vocabulary": ["<unk>", "▁hello"],
                "jointnet": {
                    "joint_hidden": 640,
                    "activation": "relu",
                    "encoder_hidden": 1_024,
                    "pred_hidden": 640,
                },
            },
        },
        "mlx_conversion": {
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "modules": {"stt_model.embed_tokens": {"group_size": 64, "bits": 4}},
            }
        },
    }


def _base_config():
    return {
        "vocab_size": 131_072,
        "hidden_size": 4_480,
        "intermediate_size": 15_680,
        "num_hidden_layers": 56,
        "max_position_embeddings": 131_072,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "mamba_num_heads": 128,
        "mamba_head_dim": 80,
        "mamba_state_dim": 128,
        "mamba_num_groups": 8,
        "conv_kernel": 4,
        "hybrid_override_pattern": "M" * 56,
    }


def test_build_runtime_config_preserves_finite_time_step_limit():
    base = _base_config()
    base["time_step_limit"] = [0.001, 0.1]

    config = build_runtime_config(_source_config(), base)

    assert config["text_config"]["time_step_limit"] == [0.001, 0.1]


def test_build_runtime_config_rejects_non_finite_time_step_limit():
    base = _base_config()
    base["time_step_limit"] = [0.0, float("inf")]

    with pytest.raises(ValueError, match="two finite numbers"):
        build_runtime_config(_source_config(), base)


def test_build_runtime_config_keeps_bf16_unquantized():
    source = _source_config()
    source["mlx_conversion"]["quantization"] = None

    config = build_runtime_config(source, _base_config())

    assert "quantization" not in config
    assert "quantization_config" not in config


def _write_artifact_inputs(tmp_path):
    source = tmp_path / "source"
    tokenizer = tmp_path / "tokenizer"
    source.mkdir()
    tokenizer.mkdir()
    (source / "config.json").write_text(json.dumps(_source_config()))
    (tokenizer / "config.json").write_text(json.dumps(_base_config()))

    shard_name = "model-00001-of-00001.safetensors"
    (source / shard_name).write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"stt_model.weight": shard_name}})
    )
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizer"})
    )
    (tokenizer / "special_tokens_map.json").write_text("{}")
    return source, tokenizer, shard_name


def test_prepare_artifact_writes_generic_load_layout(tmp_path):
    source, tokenizer, shard_name = _write_artifact_inputs(tmp_path)
    output = tmp_path / "output"

    result = convert.prepare_artifact(source, tokenizer, output, copy_weights=True)

    assert result == output
    assert json.loads((output / "config.json").read_text())["model_type"] == (
        "nemotron_voicechat"
    )
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    assert tokenizer_config["fix_mistral_regex"] is True
    assert (output / "tokenizer.json").exists()
    assert (output / "model.safetensors.index.json").exists()
    assert (output / shard_name).read_bytes() == b"weights"
    assert not (output / shard_name).is_symlink()
    assert (output / "README.md").exists()


def test_prepare_artifact_can_link_weight_shards(tmp_path):
    source, tokenizer, shard_name = _write_artifact_inputs(tmp_path)
    output = tmp_path / "linked-output"

    convert.prepare_artifact(source, tokenizer, output)

    assert (output / "model.safetensors.index.json").is_symlink()
    assert (output / shard_name).is_symlink()
