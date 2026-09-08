"""Audio output routing, token fidelity, and waveform serialization."""

import contextlib
import sys
import wave
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate import AudioGenerationResult
from mlx_vlm.generate import audio as audio_module
from mlx_vlm.generate import dispatch, generate_audio, save_audio
from mlx_vlm.generate.common import GenerationResult
from mlx_vlm.tests.test_generate import MockDetokenizer, MockModel, MockProcessor


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


def test_audio_returns_waveform_without_file(monkeypatch):
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="test"),
        generate_audio=Mock(return_value=AudioGenerationResult(audio=mx.zeros(10))),
    )
    monkeypatch.setattr(
        dispatch, "generate", Mock(return_value=GenerationResult(token_ids=[3]))
    )
    monkeypatch.setattr(
        audio_module, "wired_limit", lambda *a: contextlib.nullcontext()
    )
    result = generate_audio(
        model, SimpleNamespace(tokenizer=None), "prompt", input_ids=mx.array([[1]])
    )
    assert result.audio.shape == (10,)
    assert result.path is None


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


def test_audio_cli_uses_tts_template_and_shared_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_vlm.generate",
            "--output-modality",
            "audio",
            "--output",
            str(tmp_path / "out.wav"),
            "--ref-audio",
            "voice.wav",
            "--prompt",
            "Say hello.",
        ],
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="minicpmo"))
    monkeypatch.setattr(dispatch, "load", Mock(return_value=(model, object())))
    template = Mock(return_value="formatted")
    monkeypatch.setattr(dispatch, "apply_chat_template", template)
    speech = Mock(
        return_value=AudioGenerationResult(text="Hello.", path=tmp_path / "out.wav")
    )
    monkeypatch.setattr(dispatch, "generate_audio", speech)
    dispatch.main()
    assert template.call_args.kwargs["use_tts_template"] is True
    assert speech.call_args.args[2] == "formatted"
    assert speech.call_args.kwargs["ref_audio_path"] == "voice.wav"


@pytest.mark.parametrize(
    "flags, message",
    [
        ([], "--output is required"),
        (["--output", "out.wav", "--chat"], "does not support --chat"),
    ],
)
def test_invalid_audio_cli_fails_before_loading(monkeypatch, flags, message):
    monkeypatch.setattr(
        sys, "argv", ["mlx_vlm.generate", "--output-modality", "audio", *flags]
    )
    loader = Mock()
    monkeypatch.setattr(dispatch, "load", loader)
    with pytest.raises(ValueError, match=message):
        dispatch.main()
    loader.assert_not_called()
