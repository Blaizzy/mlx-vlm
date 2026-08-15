import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import load
from mlx_vlm.models.gemma3.config import TextConfig as Gemma3TextConfig
from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.nemotron_voicechat.config import CharacterEncoderConfig, MoGConfig
from mlx_vlm.models.nemotron_voicechat.model import Model
from mlx_vlm.models.nemotron_voicechat.session import VoiceChatSession
from mlx_vlm.models.nemotron_voicechat.streaming import (
    VoiceChatFrameTiming,
    VoiceChatProfile,
)
from mlx_vlm.models.nemotron_voicechat.tts import (
    CharAwareSubwordEncoder,
    MoGHead,
    OffsetRMSNorm,
    _top_p_logits,
)


def test_offset_rms_norm_matches_definition():
    norm = OffsetRMSNorm(4, eps=1e-6)
    norm.weight = mx.array([0.0, 0.5, -0.25, 1.0])
    inputs = mx.array([[[1.0, -2.0, 3.0, -4.0]]])
    actual = norm(inputs)
    expected = inputs / mx.sqrt(mx.mean(inputs**2, axis=-1, keepdims=True) + 1e-6)
    expected = expected * (1.0 + norm.weight)
    assert mx.allclose(actual, expected, atol=1e-6)


def test_top_p_keeps_at_least_the_largest_logit():
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
    filtered = _top_p_logits(logits, 0.01)
    assert bool(mx.isfinite(filtered[0, 3]))
    assert int(mx.sum(mx.isfinite(filtered))) == 1


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
        intermediate_size=16,
        low_rank=2,
        num_layers=1,
        num_predictions=4,
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

    created = Model.create_session(model, processor)

    assert isinstance(created, VoiceChatSession)
    assert created.model is model
    assert created.tokenizer is processor.tokenizer
    assert tts.vocabulary == vocabulary


def test_model_create_session_requires_tokenizer_interface():
    with pytest.raises(TypeError, match="tokenizer-like processor"):
        Model.create_session(SimpleNamespace(), object())


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
        audio_path,
        system_prompt="Answer briefly.",
        max_frames=2,
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
        audio,
        system_prompt="Answer briefly.",
        max_frames=2,
        seed=0,
    )
    stream = session.create_streaming_session(
        system_prompt="Answer briefly.",
        seed=0,
        max_streaming_seconds=1.0,
    )
    events = stream.push_audio(audio, sample_rate=16_000)
    audio_event = next(event for event in events if event.kind == "audio")

    assert stream._text_tokens[-1] == int(offline.text_tokens[0])
    assert stream._function_tokens[-1] == int(offline.function_tokens[0])
    assert mx.array_equal(audio_event.audio_codes, offline.audio_codes[0])
    assert mx.allclose(audio_event.samples, offline.audio[:1764], atol=2e-4)
