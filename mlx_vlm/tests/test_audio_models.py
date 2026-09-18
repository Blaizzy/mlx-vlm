"""Audio model components, generation, streaming, loading, and resampling."""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import sys
import types
import unittest
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import load
from mlx_vlm.generate import AudioGenerationResult, ar
from mlx_vlm.generate import audio as audio_module
from mlx_vlm.generate import dispatch, generate_audio, save_audio
from mlx_vlm.generate.common import GenerationResult
from mlx_vlm.models.cache import BatchKVCache, make_prompt_cache
from mlx_vlm.tests.test_generate import MockDetokenizer, MockModel, MockProcessor
from mlx_vlm.utils import load_audio


def _model_module(name):
    return importlib.import_module("mlx_vlm.models." + name)


nemotron_h = _model_module("nemotron_h.language")
nemotron = _model_module("nemotron_h_nano_omni")
nemotron_audio = _model_module("nemotron_h_nano_omni.audio")
voicechat = _model_module("nemotron_voicechat")
voicechat_config = _model_module("nemotron_voicechat.config")
voicechat_convert = _model_module("nemotron_voicechat.convert")
voicechat_tts = _model_module("nemotron_voicechat.tts")
qwen_omni = _model_module("qwen3_omni_moe")
omni_language = _model_module("qwen3_omni_moe.language")


# Audio model components


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
                    assert prompt.startswith("<|im_start|>system\nClone the voice")
                    assert prompt.endswith("<|tts_bos|>")
                    assert prompt.count("<|im_start|>system") == 1
                    assert processor._count_audio_markers(prompt) == len(paths)
                    if system:
                        assert "Be concise.<|im_end|>" in prompt
                    if input_audio:
                        assert paths == ["voice.wav", "input.wav"]
                        assert "<|im_start|>user\n<audio>Say hello." in prompt
                    else:
                        assert paths == ["voice.wav"]
                        assert prompt.endswith(user)
                    if isinstance(input_audio, list):
                        assert input_audio == ["input.wav"]
                    # A caller-supplied system reference must not be duplicated.
                    again = processor.prepare_audio_generation(
                        prompt, audio=paths, ref_audio_path="./voice.wav"
                    )
                    assert again == (prompt, paths)

    def test_audio_input_can_also_supply_reference_voice(self):
        from mlx_vlm.models.minicpmo.processing_minicpmo import MiniCPMOProcessor

        processor = MiniCPMOProcessor.__new__(MiniCPMOProcessor)
        prompt, paths = processor.prepare_audio_generation(
            "<|im_start|>user\n<audio>./</audio>Respond.<|im_end|>\n",
            audio=["input.wav"],
        )
        assert paths == ["input.wav", "input.wav"]
        assert processor._count_audio_markers(prompt) == 2
        assert "<|im_start|>user\n<audio>Respond." in prompt

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
        assert out.new_ids.shape == (1, 2, 1)

    def test_sanitize_materializes_tts_weight_norm(self):
        from mlx_vlm.models.minicpmo.config import (
            MiniCPMTTSConfig,
            ModelConfig,
            TextConfig,
            VisionConfig,
        )
        from mlx_vlm.models.minicpmo.minicpmo import Model

        text = _small_config(
            TextConfig,
            model_type="qwen3",
            hidden_size=8,
            intermediate_size=16,
            rms_norm_eps=1e-6,
            vocab_size=20,
            num_key_value_heads=2,
            head_dim=4,
            rope_theta=10000,
            max_position_embeddings=64,
        )
        vision = _small_config(
            VisionConfig,
            hidden_size=8,
            intermediate_size=16,
        )
        tts = _small_config(
            MiniCPMTTSConfig,
            hidden_size=8,
            intermediate_size=16,
            num_key_value_heads=2,
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

        assert "tts.head_code.0.weight" in sanitized
        assert sanitized["tts.head_code.0.weight"].shape == (12, 8)

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

        assert processor.tokenizer.tts_start_id == 12
        assert processor.tokenizer.tts_end_id == 13

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
        assert output.audio_tokens.shape == (1, 1, 1)
        assert captured["tts_max_new_token"] == 5
        assert captured["tts_bound"] == (2, 3)
        assert captured["mask"].shape == (1, 3)
        assert captured["max_tokens"] == 99
        params = captured["tts_sampling_params"]
        assert params.temperature == 0.2
        assert params.top_p == 0.3
        assert params.top_k == 6
        assert params.repetition_penalty == 1.2

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

        assert calls == [((), {})]


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
        qwen_omni.TextConfig,
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
    return qwen_omni.VisionConfig(
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
    thinker_config = qwen_omni.ThinkerConfig(
        text_config=text_config,
        vision_config=vision_config or _qwen_vision_config(),
        audio_config=qwen_omni.AudioConfig(
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
    talker_config = qwen_omni.TalkerConfig(
        text_config=_tiny_text_config("qwen3_omni_moe_talker_text"),
        code_predictor_config=_small_config(
            qwen_omni.CodePredictorConfig,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            num_code_groups=2,
        ),
        accept_hidden_layer=0,
        thinker_hidden_size=16,
    )
    code2wav_config = _small_config(
        qwen_omni.Code2WavConfig,
        num_key_value_heads=2,
        decoder_dim=16,
        codebook_dim=8,
        codebook_size=32,
        num_quantizers=2,
        num_semantic_quantizers=1,
        semantic_codebook_size=32,
        vector_quantization_hidden_dimension=8,
    )
    return qwen_omni.Model(
        qwen_omni.ModelConfig(
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

        assert sequences.shape[1] == input_ids.shape[1] + 3
        assert hidden_states.shape == expected_hidden_states.shape
        assert input_embeds.shape == expected_input_embeds.shape
        assert bool(
            mx.allclose(
                hidden_states, expected_hidden_states, rtol=0.0001, atol=0.0001
            ).item()
        )
        assert bool(
            mx.allclose(
                input_embeds, expected_input_embeds, rtol=1e-06, atol=1e-06
            ).item()
        )

    def _thinker_logits(self, model, ids, cache, **kw):
        x = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
        out = model.thinker(x, cache=cache, **kw)
        return out.logits if hasattr(out, "logits") else out

    def _assert_prefill_decode_match(self, pre, step, pos):
        a, b = pre.reshape(-1), step[:, -1].reshape(-1)
        cosine = float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-9))
        assert cosine > 0.999, f"decode diverges from prefill at pos {pos}"

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
        assert predicate is not None
        assert predicate("thinker.language_model.model.layers.0.mlp.gate", None) == {
            "group_size": 64,
            "bits": 8,
        }
        assert predicate("thinker.language_model.model.layers.0.self_attn.q_proj", None)


def tiny_text_config(hidden_size=24):
    return nemotron.TextConfig(
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
    model = nemotron_h.NemotronHModel(config, with_embeddings=False)
    inputs_embeds = mx.zeros((1, 2, config.hidden_size))

    output = model(inputs_embeds=inputs_embeds)

    assert output.shape == inputs_embeds.shape
    with pytest.raises(ValueError, match="no token embedding"):
        model(mx.array([[1, 2]], dtype=mx.int32))


def tiny_vision_config():
    return _small_config(
        nemotron.VisionConfig,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        max_resolution=32,
        args={"teachers": [{"name": "test"}]},
    )


def tiny_sound_config(hidden_size=16):
    return nemotron.AudioConfig(
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
    config = nemotron.AudioConfig()
    extractor = nemotron_audio.SoundFeatureExtractor(config)
    waveform = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    features, mask, lengths = extractor([waveform])

    assert features.shape == (1, 11, config.num_mel_bins)
    assert mask.shape == (1, 11)
    assert lengths.tolist() == [11]
    assert mask.sum(axis=1).tolist() == [10]
    assert np.isfinite(np.array(features)).all()


@pytest.fixture
def nemotron_audio_model():
    model = nemotron.Model(
        nemotron.ModelConfig(
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
    return model


@pytest.mark.parametrize("scale_dtype", [mx.uint8, mx.uint32], ids=["uint8", "uint32"])
def test_audio_features_use_float_compute_dtype(nemotron_audio_model, scale_dtype):
    # Packed scales must not turn audio features into integer convolution inputs.
    model = nemotron_audio_model
    model.language_model.lm_head.scales = mx.zeros((1,), dtype=scale_dtype)

    output = model.get_input_embeddings(
        mx.array([[1, 99, 99, 99, 2]]),
        input_features=mx.random.normal((1, 17, 16)),
        feature_attention_mask=mx.ones((1, 17), dtype=mx.int32),
    )
    assert np.isfinite(np.array(output.inputs_embeds)).all()


def test_model_rejects_sound_token_feature_count_mismatch(nemotron_audio_model):
    with pytest.raises(ValueError, match="Sound token count"):
        nemotron_audio_model.get_input_embeddings(
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

    audio_sanitized = nemotron_audio.sanitize_audio_weights(weights)
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

    model = nemotron.Model(
        nemotron.ModelConfig(
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
    profile = voicechat.VoiceChatProfile(
        frame_duration_ms=80.0,
        frames=[
            voicechat.VoiceChatFrameTiming(0, 10, 2, 20, 30, 8, 72),
            voicechat.VoiceChatFrameTiming(1, 8, 1, 16, 24, 6, 56),
            voicechat.VoiceChatFrameTiming(2, 9, 1, 18, 27, 7, 63),
        ],
    )
    summary = profile.summary(drop_first=1)

    assert summary["frames"] == 2
    assert summary["dropped_cold_frames"] == 1
    assert summary["stages"]["total"]["mean_ms"] == 59.5
    assert summary["processing_frames_per_second"] == pytest.approx(1000 / 59.5)
    assert summary["realtime_factor"] == pytest.approx(59.5 / 80)


def test_character_aware_encoder_prepares_and_scatters_subwords():
    config = voicechat_config.CharacterEncoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        char_vocab_size=4,
    )
    encoder = voicechat_tts.CharAwareSubwordEncoder(config, out_size=8, vocab_size=6)
    encoder.set_vocabulary({"a": 0, "b": 1, "c": 2, "ab": 3, "<s>": 4, "</s>": 5})
    ids = mx.array([[3, 4, 5]], dtype=mx.int32)
    mask = mx.array([[True, False, True]])
    output = encoder(ids, mask)
    mx.eval(output)
    assert output.shape == (1, 3, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_mog_head_inference_shapes_and_finite_values():
    config = voicechat_config.MoGConfig(
        intermediate_size=16, low_rank=2, num_layers=1, num_predictions=4
    )
    head = voicechat_tts.MoGHead(hidden_size=8, out_size=4, config=config)
    inputs = mx.zeros((2, 1, 8))
    mean, logs = head.infer(inputs, guidance_scale=0.2, top_p=0.95)
    mx.eval(mean, logs)
    assert mean.shape == (1, 1, 4)
    assert logs.shape == (1, 1, 1)
    assert bool(mx.all(mx.isfinite(mean)))
    assert bool(mx.all(mx.isfinite(logs)))


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

    created = voicechat.Model.create_session(model, processor)

    assert isinstance(created, voicechat.VoiceChatSession)
    assert created.model is model
    assert created.tokenizer is processor.tokenizer
    assert tts.vocabulary == vocabulary


def test_model_create_session_requires_tokenizer_interface():
    with pytest.raises(TypeError, match="tokenizer-like processor"):
        voicechat.Model.create_session(SimpleNamespace(), object())


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


@pytest.mark.parametrize(
    "limit", [[0.001, 0.1], [0.0, float("inf")]], ids=["finite", "non-finite"]
)
def test_build_runtime_config_time_step_limit(limit):
    base = _base_config() | {"time_step_limit": limit}
    if not math.isfinite(limit[1]):
        with pytest.raises(ValueError, match="two finite numbers"):
            voicechat_convert.build_runtime_config(_source_config(), base)
    else:
        assert (
            voicechat_convert.build_runtime_config(_source_config(), base)[
                "text_config"
            ]["time_step_limit"]
            == limit
        )


def test_build_runtime_config_keeps_bf16_unquantized():
    source = _source_config()
    source["mlx_conversion"]["quantization"] = None

    config = voicechat_convert.build_runtime_config(source, _base_config())

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


@pytest.mark.parametrize("copy_weights", [False, True], ids=["link", "copy"])
def test_prepare_artifact_layout(tmp_path, copy_weights):
    source, tokenizer, shard = _write_artifact_inputs(tmp_path)
    output = tmp_path / ("output" if copy_weights else "linked-output")
    result = voicechat_convert.prepare_artifact(
        source, tokenizer, output, copy_weights=copy_weights
    )
    assert result == output
    assert (output / shard).is_symlink() == (not copy_weights)
    assert (output / "model.safetensors.index.json").exists()
    if not copy_weights:
        assert (output / "model.safetensors.index.json").is_symlink()
    config = json.loads((output / "config.json").read_text())
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    assert config["model_type"] == "nemotron_voicechat"
    assert tokenizer_config["fix_mistral_regex"] is True
    assert (output / "tokenizer.json").exists()
    assert (output / shard).read_bytes() == b"weights"
    assert (output / "README.md").exists()


def test_streaming_session_buffers_arbitrary_chunk_boundaries():
    stream = voicechat.VoiceChatStreamingSession.__new__(
        voicechat.VoiceChatStreamingSession
    )
    stream._closed = False
    stream.input_sample_rate = 16000
    stream.frame_samples = 4
    stream._pending_audio = mx.zeros((0,), dtype=mx.float32)
    seen = []

    def step(frame):
        seen.append(frame.tolist())
        return []

    stream._step_audio_frame = step
    assert stream.push_audio([0.0], sample_rate=16000) == []
    assert stream.push_audio([1.0, 2.0, 3.0, 4.0], sample_rate=16000) == []
    assert stream.push_audio([5.0, 6.0], sample_rate=16000) == []
    assert seen == [[0.0, 1.0, 2.0, 3.0]]
    assert stream._pending_audio.tolist() == [4.0, 5.0, 6.0]


# Audio generation and input loading

# Generation


def test_reference_conditions_both_generation_stages(monkeypatch):
    from mlx_vlm.models.minicpmo.processing_minicpmo import MiniCPMOProcessor

    processor = MiniCPMOProcessor.__new__(MiniCPMOProcessor)
    processor.tokenizer = object()
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="minicpmo"),
        generate_audio=Mock(return_value=AudioGenerationResult(audio=mx.zeros(10))),
    )
    features = [mx.ones((1, 80, 12))]
    prepare = Mock(
        return_value=dict(
            input_ids=mx.array([[1, 2, 3, 4]]),
            attention_mask=mx.ones((1, 4)),
            audio_features=features,
            audio_bounds=[[(1, 2), (2, 3)]],
        )
    )
    text_generate = Mock(return_value=GenerationResult(token_ids=[5]))
    monkeypatch.setattr(dispatch, "prepare_inputs", prepare)
    monkeypatch.setattr(dispatch, "generate", text_generate)
    monkeypatch.setattr(
        audio_module, "wired_limit", lambda *a: contextlib.nullcontext()
    )
    generate_audio(
        model,
        processor,
        "<|im_start|>user\nDescribe this audio.<|im_end|>\n<|im_start|>assistant\n<|tts_bos|>",
        audio=["input.wav"],
        ref_audio_path="voice.wav",
    )
    assert prepare.call_args.kwargs["audio"] == ["voice.wav", "input.wav"]
    prompt = prepare.call_args.kwargs["prompts"]
    assert prompt.startswith("<|im_start|>system\nClone the voice")
    assert "<|im_start|>user\n<audio>Describe this audio." in prompt
    for call in (text_generate.call_args, model.generate_audio.call_args):
        assert call.kwargs["audio_features"] is features
        assert call.kwargs["audio_bounds"] == [[(1, 2), (2, 3)]]
    assert model.generate_audio.call_args.kwargs["ref_audio_path"] == "voice.wav"


def test_precomputed_inputs_cannot_silently_skip_reference_conditioning(monkeypatch):
    processor = SimpleNamespace(prepare_audio_generation=Mock())
    model = SimpleNamespace(generate_audio=Mock())
    text_generate = Mock()
    monkeypatch.setattr(dispatch, "generate", text_generate)
    with pytest.raises(ValueError, match="not precomputed input_ids"):
        generate_audio(model, processor, "prompt", input_ids=mx.array([[1]]))
    processor.prepare_audio_generation.assert_not_called()
    text_generate.assert_not_called()


@pytest.mark.parametrize("multiple_audio", [False, True])
def test_prepare_inputs_preserves_audio_for_capable_processors(
    monkeypatch, multiple_audio
):
    from mlx_vlm import utils

    processor = SimpleNamespace(supports_multiple_audio=multiple_audio)
    load = Mock(
        side_effect=lambda path, sr: (
            np.ones(16) if path == "voice.wav" else np.zeros(16)
        )
    )
    process = Mock(return_value={"input_ids": mx.array([[1]])})
    monkeypatch.setattr(utils, "load_audio", load)
    monkeypatch.setattr(utils, "process_inputs_with_fallback", process)
    utils.prepare_inputs(processor, prompts="prompt", audio=["voice.wav", "input.wav"])
    audios = process.call_args.kwargs["audio"]
    assert len(audios) == (2 if multiple_audio else 1)
    np.testing.assert_array_equal(audios[0], np.ones(16))
    if multiple_audio:
        np.testing.assert_array_equal(audios[1], np.zeros(16))


@pytest.mark.parametrize("tokens", [[], [7], [7, 8], [7, 2], [2]])
def test_final_token_ids_preserve_eos_without_duplicate_flush(monkeypatch, tokens):
    model, processor = MockModel(), MockProcessor()
    monkeypatch.setattr(
        dispatch,
        "generate_step",
        lambda *a, **kw: iter((t, mx.zeros(4)) for t in tokens),
    )
    monkeypatch.setattr(
        dispatch, "make_streaming_detokenizer", lambda p: MockDetokenizer()
    )
    monkeypatch.setattr(dispatch, "wired_limit", lambda *a: contextlib.nullcontext())
    results = list(
        dispatch.stream_generate(
            model, processor, "prompt", input_ids=mx.array([[1, 3]])
        )
    )
    assert results[-1].token_ids == tokens
    assert results[-1].generation_tokens == len(tokens)
    assert all(r.token_ids is None for r in results[:-1])


@pytest.mark.parametrize("precomputed", [False, True])
def test_audio_uses_shared_text_generation_and_prepares_once(
    monkeypatch, precomputed, tmp_path
):
    model = SimpleNamespace(config=SimpleNamespace(model_type="minicpmo"))
    model.generate_audio = Mock(
        return_value=AudioGenerationResult(
            audio=mx.zeros(240), audio_tokens=mx.array([[[5]]])
        )
    )
    processor = SimpleNamespace(tokenizer=object())
    inputs = dict(
        input_ids=mx.array([[1, 2]]),
        pixel_values=["image"],
        attention_mask=mx.ones((1, 2)),
        audio_bounds=[(1, 2)],
    )
    prepare = Mock(return_value=inputs)
    monkeypatch.setattr(dispatch, "prepare_inputs", prepare)
    text_generate = Mock(
        return_value=GenerationResult(
            text="hello",
            token_ids=[3, 4],
            prompt_tokens=2,
            generation_tokens=2,
            finish_reason="stop",
        )
    )
    monkeypatch.setattr(dispatch, "generate", text_generate)
    monkeypatch.setattr(
        audio_module, "wired_limit", lambda *a: contextlib.nullcontext()
    )
    kwargs = dict(temperature=0.1, tts_temperature=0.8, tts_max_tokens=20)
    if precomputed:
        kwargs.update(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            mask=inputs["attention_mask"],
            audio_bounds=inputs["audio_bounds"],
        )
    result = generate_audio(
        model,
        processor,
        "prompt",
        image=["image.png"],
        audio=["input.wav"],
        ref_audio_path="voice.wav",
        output_audio_path=tmp_path / "out.wav",
        **kwargs,
    )
    assert prepare.call_count == (0 if precomputed else 1)
    assert result.text == "hello"
    assert result.token_ids == [3, 4]
    assert result.prompt_tokens == 2
    assert result.finish_reason == "stop"
    assert result.path.is_file()
    text_kwargs = text_generate.call_args.kwargs
    assert text_kwargs["temperature"] == 0.1
    assert "tts_temperature" not in text_kwargs
    speech_kwargs = model.generate_audio.call_args.kwargs
    assert speech_kwargs["generated_tokens"] == [3, 4]
    assert speech_kwargs["tts_temperature"] == 0.8
    assert speech_kwargs["tts_max_tokens"] == 20
    assert speech_kwargs["audio_bounds"] == [(1, 2)]
    assert speech_kwargs["ref_audio_path"] == "voice.wav"


def test_unsupported_audio_fails_before_text_generation(monkeypatch):
    text_generate = Mock()
    monkeypatch.setattr(dispatch, "generate", text_generate)
    with pytest.raises(ValueError, match="TTS module"):
        generate_audio(SimpleNamespace(), None, "prompt")
    text_generate.assert_not_called()


def test_save_audio_writes_valid_wav(tmp_path):
    result = AudioGenerationResult(audio=mx.sin(mx.arange(2400) * 0.1) * 0.1)
    path = save_audio(result, tmp_path / "nested" / "speech.wav")
    with wave.open(BytesIO(path.read_bytes())) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == result.sample_rate
        assert wav.getnframes() == 2400
    with pytest.raises(ValueError, match=".wav"):
        save_audio(result, tmp_path / "speech.mp3")


# Audio loading and resampling


def test_load_audio_uses_mlx_audio_io_and_returns_mono_float32(monkeypatch):
    from mlx_audio import audio_io

    calls = []

    def fake_read(file, dtype="float64"):
        calls.append((file, dtype))
        return (
            np.array([[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]], dtype=np.float32),
            16000,
        )

    monkeypatch.setattr(audio_io, "read", fake_read)

    audio = load_audio("sample.wav", sr=16000)

    assert calls == [("sample.wav", "float32")]
    assert audio.dtype == np.float32
    assert audio.shape == (3,)
    np.testing.assert_allclose(audio, np.array([0.5, 0.0, 0.5], dtype=np.float32))


def test_load_audio_downmixes_stereo_before_resampling(monkeypatch):
    from mlx_audio import audio_io
    from mlx_audio.utils import resample_audio

    rng = np.random.default_rng(0)
    stereo = rng.standard_normal((48000, 2)).astype(np.float32)

    monkeypatch.setattr(audio_io, "read", lambda file, dtype="float64": (stereo, 48000))

    audio = load_audio("stereo_48k.wav", sr=16000)

    expected = np.asarray(resample_audio(stereo.mean(axis=1), 48000, 16000))
    assert audio.dtype == np.float32
    assert audio.shape == (16000,)
    np.testing.assert_allclose(audio, expected, rtol=1e-5, atol=1e-6)


def _host(a):
    return np.array(a.astype(mx.float32))


def _padded_image_batch(model):
    rows = [
        [1] * prefix + [VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END, 2, 3, 4]
        for prefix in [8, 3]
    ]
    lengths = [len(row) for row in rows]
    padding = [max(lengths) - length for length in lengths]
    ids = mx.array([[0] * p + row for p, row in zip(padding, rows)])
    mask = mx.array([[0] * p + [1] * len(row) for p, row in zip(padding, rows)])
    kw = model.get_input_embeddings(
        ids,
        pixel_values=mx.random.normal((32, 24)),
        image_grid_thw=mx.array([[1, 4, 4], [1, 4, 4]]),
        mask=mask,
    ).to_dict()
    return ids, mask, padding, kw


@pytest.mark.parametrize("expanded", [False, True])
def test_omni_full_masks_use_physical_padded_cache_columns(expanded):
    mx.random.seed(17)
    model = _tiny_vision_model()
    ids, _, padding, kw = _padded_image_batch(model)
    embeds = kw.pop("inputs_embeds")
    positions = kw.pop("position_ids")
    if expanded:
        kw["visual_pos_masks"] = kw["visual_pos_masks"][..., None]
    lm = model.language_model

    def caches():
        return [BatchKVCache(padding) for _ in lm.layers]

    full = lm(
        ids, inputs_embeds=embeds, position_ids=positions, cache=caches(), **kw
    ).logits
    cache = caches()
    chunks = [
        lm(
            ids[:, start:stop],
            inputs_embeds=embeds[:, start:stop],
            position_ids=positions[..., start:stop],
            cache=cache,
            **kw,
        ).logits
        for start, stop in [(0, 11), (11, ids.shape[1])]
    ]
    got = mx.concatenate(chunks, axis=1)
    for row, pad in enumerate(padding):
        np.testing.assert_allclose(
            _host(got[row, pad:]), _host(full[row, pad:]), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("batched", [False, True])
def test_omni_prompt_scheduler_keeps_mask_and_features_paired(batched):
    mx.random.seed(17)
    model = _tiny_vision_model()
    lm = model.language_model
    if batched:
        ids, attention_mask, padding, kw = _padded_image_batch(model)
    else:
        ids, pixels, grid = _image_inputs()
        kw = model.get_input_embeddings(
            ids, pixel_values=pixels, image_grid_thw=grid
        ).to_dict()
        attention_mask = mx.ones_like(ids)
        padding = [0]
    full = lm(ids, cache=[BatchKVCache(padding) for _ in lm.layers], **kw).logits
    batch_size = ids.shape[0]
    rows = ar._split_prompt_kwargs_per_row(kw, batch_size)
    prompts, rows = ar._unpad_batch_prompts(ids, attention_mask, rows)
    embeds, prompt_kwargs = ar._merge_prefill_prompt_kwargs(rows, prompts)
    batch = ar.PromptProcessingBatch(
        model=lm,
        uids=list(range(batch_size)),
        input_ids=prompts,
        max_tokens=[1] * batch_size,
        inputs_embeds=embeds,
        prompt_kwargs=prompt_kwargs,
        prefill_step_size=5,
    )
    outputs = []
    original = omni_language.LanguageModel.__call__

    def recording(self, *args, **kwargs):
        # Each call receives the residuals for its input columns, with zeros
        # at text and padding positions.
        start = sum(o.shape[1] for o in outputs)
        stop = start + args[0].shape[1]
        np.testing.assert_array_equal(
            _host(kwargs["deepstack_visual_embeds"]),
            _host(kw["deepstack_visual_embeds"][:, start:stop]),
        )
        result = original(self, *args, **kwargs)
        outputs.append(result.logits)
        return result

    with patch.object(omni_language.LanguageModel, "__call__", recording):
        while batch.needs_processing():
            batch.prompt_step()
        batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    got = mx.concatenate(outputs, axis=1)
    for row, pad in enumerate(padding):
        # Compare multi-token logits directly; use cosine similarity for the
        # final token to allow rounding differences between attention kernels.
        np.testing.assert_allclose(
            _host(got[row, pad:-1]), _host(full[row, pad:-1]), rtol=1e-4, atol=1e-4
        )
        a, b = _host(got[row, -1]), _host(full[row, -1])
        assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.999


def test_cold_batch_merge_keeps_features_when_text_row_is_first():
    model = _tiny_vision_model()
    ids, pixels, grid = _image_inputs()
    text = model.get_input_embeddings(mx.array([[1, 2]])).to_dict()
    image = model.get_input_embeddings(
        ids, pixel_values=pixels, image_grid_thw=grid
    ).to_dict()
    assert text["deepstack_visual_embeds"].shape == (1, 2, 2, 16)
    assert not mx.any(text["deepstack_visual_embeds"]).item()
    embeds, kw = ar._merge_prefill_prompt_kwargs(
        [text, image], [[1, 2], ids[0].tolist()]
    )
    assert kw["deepstack_visual_embeds"].shape == (2, ids.shape[1], 2, 16)
    assert not mx.any(kw["deepstack_visual_embeds"][0]).item()
    np.testing.assert_array_equal(
        _host(kw["deepstack_visual_embeds"][1:]),
        _host(image["deepstack_visual_embeds"]),
    )
    lm = model.language_model
    reference = lm(ids, cache=make_prompt_cache(lm), **image).logits[:, -1]
    batch = ar.PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=[[1, 2], ids[0].tolist()],
        max_tokens=[1, 1],
        inputs_embeds=embeds,
        prompt_kwargs=kw,
        prefill_step_size=5,
    )
    while batch.needs_processing():
        batch.prompt_step()
    result = batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    assert result._next_tokens[1:].tolist() == mx.argmax(reference, axis=-1).tolist()


def _mixed_batch(model, rows, prompts, prefix, warm_cache=None):
    bg = object.__new__(ar.BatchGenerator)
    bg.model = model
    bg.apc_manager = object()
    bg.apc = SimpleNamespace(
        prepare_prefill=lambda *a, **k: None,
        observe_cache=lambda *a, **k: None,
        merge_rows=lambda *a, **k: (
            ar._apc.make_warm_batch_exact_cache_multi(
                [warm_cache, make_prompt_cache(model)], [prefix, 0]
            )
            if warm_cache is not None
            else ([], prefix)
        ),
        checkpoint_len=lambda *a: 0,
        checkpoint_lengths=lambda *a: [],
    )
    bg.apc_mode = "exact"
    bg.prefill_step_size = 2
    bg.kv_bits = None
    bg.kv_group_size = 64
    bg.kv_quant_scheme = "affine"
    bg._wire_stack = None
    sequences = [
        (i, ids, 1, kw, [], None) for i, (ids, kw) in enumerate(zip(prompts, rows))
    ]
    with patch.object(
        bg, "_apc_pick_for", side_effect=[{"prefix_len": prefix, "extra_hash": 0}, None]
    ):
        batch = bg._build_mixed_prompt_batch(sequences)
    # Limit this fixture to cache restoration and prompt processing.
    batch._apc_meta = []
    batch._apc_manager = None
    return batch


def test_mixed_cached_prefix_slices_visual_features_before_merging():
    # The warm row's cached prefix contains one visual token; the cold row
    # contains two visual tokens.
    calls = []

    def model(ids, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(logits=mx.zeros((*ids.shape, 4)))

    rows = [
        {
            "inputs_embeds": mx.zeros((1, 6, 4)),
            "deepstack_visual_embeds": mx.array([0, 10, 0, 11, 12, 0]).reshape(
                1, 6, 1, 1
            ),
        },
        {
            "inputs_embeds": mx.zeros((1, 4, 4)),
            "deepstack_visual_embeds": mx.array([20, 0, 21, 0]).reshape(1, 4, 1, 1),
        },
    ]
    batch = _mixed_batch(model, rows, [list(range(6)), list(range(4))], 3)
    assert batch._prompt_kwargs["deepstack_visual_embeds"][:, :, 0, 0].tolist() == [
        [11, 12, 0, 0],
        [20, 0, 21, 0],
    ]
    while batch.needs_processing():
        batch.prompt_step()
    batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    assert [c["deepstack_visual_embeds"][:, :, 0, 0].tolist() for c in calls] == [
        [[11, 12], [20, 0]],
        [[0], [21]],
        [[0], [0]],
    ]


@pytest.mark.parametrize("prefix", [2, 11, 14])
def test_omni_mixed_warm_and_cold_cache_matches_full_prompt(prefix):
    mx.random.seed(17)
    model = _tiny_vision_model()
    ids, attention_mask, _, kw = _padded_image_batch(model)
    lm = model.language_model
    rows = ar._split_prompt_kwargs_per_row(kw, 2)
    prompts, rows = ar._unpad_batch_prompts(ids, attention_mask, rows)
    reference = mx.concatenate(
        [
            lm(mx.array([p]), cache=make_prompt_cache(lm), **r).logits[:, -1]
            for p, r in zip(prompts, rows)
        ]
    )
    warm_cache = make_prompt_cache(lm)
    warm_kwargs = dict(rows[0])
    warm_kwargs["inputs_embeds"] = warm_kwargs["inputs_embeds"][:, :prefix]
    lm(mx.array([prompts[0][:prefix]]), cache=warm_cache, **warm_kwargs)
    batch = _mixed_batch(lm, rows, prompts, prefix, warm_cache)
    while batch.needs_processing():
        batch.prompt_step()
    sampled_logprobs = []

    def sample(logprobs):
        sampled_logprobs.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    result = batch.generate(sample, lambda _: False)
    assert result._next_tokens.tolist() == mx.argmax(reference, axis=-1).tolist()
    expected = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        _host(sampled_logprobs[0]), _host(expected), atol=0.005, rtol=0.001
    )
