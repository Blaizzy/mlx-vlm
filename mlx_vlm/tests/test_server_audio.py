import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.audio as server_audio


@pytest.fixture
def client(reset_audio_runtime):
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_audio_runtime(monkeypatch):
    if server.runtime.audio_queue is not None:
        server.runtime.audio_queue.stop_and_join()
    os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
    os.environ.pop("MLX_VLM_PRELOAD_IMAGE_MODEL", None)
    os.environ.pop("MLX_VLM_PRELOAD_TTS_MODEL", None)
    os.environ.pop("MLX_VLM_PRELOAD_STT_MODEL", None)
    monkeypatch.setattr(server.runtime, "audio_queue", None)
    monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    yield
    if server.runtime.audio_queue is not None:
        server.runtime.audio_queue.stop_and_join()
        server.runtime.audio_queue = None


def _fake_audio_write(target, audio, sample_rate, format="wav"):
    payload = f"{format}:{sample_rate}:{np.array(audio).shape[0]}".encode()
    if hasattr(target, "write"):
        target.write(payload)
    else:
        Path(target).write_bytes(payload)


class FakeTTSModel:
    model_type = "fake_tts"
    sample_rate = 16000

    def __init__(self):
        self.calls = []

    def generate(self, text, voice=None, speed=None, stream=False, **kwargs):
        self.calls.append(
            {"text": text, "voice": voice, "speed": speed, "stream": stream, **kwargs}
        )
        yield SimpleNamespace(
            audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=self.sample_rate,
        )


def test_audio_speech_returns_audio_bytes(client, monkeypatch):
    fake_model = FakeTTSModel()
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return fake_model, None, SimpleNamespace(model_type="audio")

    monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "fake-tts",
            "input": "Hello world",
            "voice": "alloy",
            "speed": 1.25,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/wav"
    assert (
        response.headers["content-disposition"].lower()
        == "attachment; filename=speech.wav"
    )
    assert response.content == b"wav:16000:3"
    assert fake_model.calls[0]["text"] == "Hello world"
    assert fake_model.calls[0]["voice"] == "alloy"
    assert fake_model.calls[0]["speed"] == 1.25
    assert cache_calls == [("fake-tts", {"model_kind": "audio_tts"})]

    metrics = client.get("/metrics").json()
    assert metrics["latest"]["endpoint"] == "/v1/audio/speech"
    assert metrics["latest"]["backend"] == "audio_queue"


def test_audio_speech_stream_bad_model_returns_error_before_headers(
    client, monkeypatch
):
    def raise_not_found(model, **kwargs):
        raise HTTPException(status_code=404, detail=f"missing {model}")

    monkeypatch.setattr(server, "get_cached_model", raise_not_found)

    response = client.post(
        "/v1/audio/speech",
        json={"model": "missing-model", "input": "hi", "stream": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "missing missing-model"


class FakeSTTModel:
    model_type = "fake_stt"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, path, context=None, language=None, task=None, **kwargs):
        self.calls.append(
            {
                "path": path,
                "context": context,
                "language": language,
                "task": task,
                **kwargs,
            }
        )
        assert Path(path).exists()
        return self.result


def test_audio_transcriptions_default_json(client, monkeypatch):
    fake_model = FakeSTTModel({"text": "This is a test transcription."})
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return fake_model, None, SimpleNamespace(model_type="audio")

    monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
    monkeypatch.setattr(
        server_audio,
        "audio_read",
        lambda buffer, always_2d=False: (np.zeros(160, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"audio-bytes", "audio/wav")},
        data={"model": "fake-stt", "prompt": "prior context", "language": "en"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "This is a test transcription."}
    assert fake_model.calls[0]["context"] == "prior context"
    assert fake_model.calls[0]["language"] == "en"
    assert cache_calls == [("fake-stt", {"model_kind": "audio_stt"})]


def test_audio_transcriptions_text_response_format(client, monkeypatch):
    fake_model = FakeSTTModel({"text": "Plain text transcript."})
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (fake_model, None, SimpleNamespace(model_type="audio")),
    )
    monkeypatch.setattr(
        server_audio,
        "audio_read",
        lambda buffer, always_2d=False: (np.zeros(160, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"audio-bytes", "audio/wav")},
        data={"model": "fake-stt", "response_format": "text"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Plain text transcript."


def test_audio_translations_passes_translate_task(client, monkeypatch):
    fake_model = FakeSTTModel({"text": "Translated transcript."})
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (fake_model, None, SimpleNamespace(model_type="audio")),
    )
    monkeypatch.setattr(
        server_audio,
        "audio_read",
        lambda buffer, always_2d=False: (np.zeros(160, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)

    response = client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", b"audio-bytes", "audio/wav")},
        data={"model": "fake-stt"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Translated transcript."}
    assert fake_model.calls[0]["task"] == "translate"


def test_audio_tts_and_stt_caches_are_independent(monkeypatch):
    fake_tts = SimpleNamespace(model_type="fake_tts")
    fake_stt = SimpleNamespace(model_type="fake_stt")

    def fake_load_audio_model(model_path):
        return {"fake-tts": fake_tts, "fake-stt": fake_stt}[model_path]

    monkeypatch.setattr(server, "load_audio_model", fake_load_audio_model)

    tts_model, _, _ = server.get_cached_model("fake-tts", model_kind="audio_tts")
    stt_model, _, _ = server.get_cached_model("fake-stt", model_kind="audio_stt")

    assert tts_model is fake_tts
    assert stt_model is fake_stt
    assert server.runtime.model_cache.for_kind("tts")["model"] is fake_tts
    assert server.runtime.model_cache.for_kind("stt")["model"] is fake_stt

    cached_tts_model, _, _ = server.get_cached_model("fake-tts", model_kind="audio_tts")

    assert cached_tts_model is fake_tts


def test_audio_transcriptions_undecodable_upload_returns_400(client, monkeypatch):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: pytest.fail("model must not load for a bad upload"),
    )

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("broken.m4a", b"not-audio", "audio/mp4")},
        data={"model": "fake-stt"},
    )

    assert response.status_code == 400
    assert "Invalid audio file" in response.json()["detail"]
