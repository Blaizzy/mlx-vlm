import os
import time
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


def test_audio_request_queue_serializes_requests(monkeypatch):
    active = 0
    max_active = 0
    fake_model = FakeTTSModel()

    def slow_generate(text, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.05)
        yield SimpleNamespace(
            audio=np.array([1.0], dtype=np.float32), sample_rate=16000
        )
        active -= 1

    fake_model.generate = slow_generate
    monkeypatch.setattr(
        server_audio,
        "get_cached_model",
        lambda model, **kwargs: (fake_model, None, SimpleNamespace(model_type="audio")),
    )
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)

    audio_queue = server_audio.AudioRequestQueue()
    try:
        first = audio_queue.submit(
            kind="tts",
            model_name="fake",
            payload=server_audio.SpeechTaskPayload(
                request=server_audio.AudioSpeechRequest(model="fake", input="first")
            ),
        )
        second = audio_queue.submit(
            kind="tts",
            model_name="fake",
            payload=server_audio.SpeechTaskPayload(
                request=server_audio.AudioSpeechRequest(model="fake", input="second")
            ),
        )

        assert _drain(first) == [b"mp3:16000:1"]
        assert _drain(second) == [b"mp3:16000:1"]
        assert max_active == 1
    finally:
        audio_queue.stop_and_join()


def test_get_cached_model_loads_tts_audio_model(monkeypatch):
    fake_model = SimpleNamespace(model_type="fake_audio")
    monkeypatch.setattr(server, "load_audio_model", lambda model_path: fake_model)

    model, processor, config = server.get_cached_model(
        "fake-audio", model_kind="audio_tts"
    )

    assert model is fake_model
    assert processor is None
    assert config.model_type == "fake_audio"
    assert server.runtime.response_generator is None
    assert server.runtime.model_cache.for_kind("tts")["model_kind"] == "audio_tts"


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


def test_audio_cache_does_not_evict_text_cache(monkeypatch):
    class FakeResponseGenerator:
        def __init__(self, model_path, adapter_path=None, **kwargs):
            self.model_path = model_path
            self.adapter_path = adapter_path
            self.model = SimpleNamespace(kind="text")
            self.processor = SimpleNamespace()
            self.config = SimpleNamespace(model_type="text")
            self.stopped = False

        def wait_until_ready(self):
            return self.model, self.processor, self.config

        def stop_and_join(self):
            self.stopped = True

    fake_audio = SimpleNamespace(model_type="fake_audio")

    monkeypatch.setattr(server._app_module, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    monkeypatch.setattr(server, "load_audio_model", lambda model_path: fake_audio)

    text_model, _, _ = server.get_cached_model("fake-text")
    text_generator = server.runtime.response_generator
    audio_model, _, _ = server.get_cached_model("fake-audio", model_kind="audio_tts")

    assert audio_model is fake_audio
    assert server.runtime.model_cache.for_kind("text_generation")["model"] is text_model
    assert server.runtime.model_cache.for_kind("tts")["model"] is fake_audio
    assert text_generator.stopped is False

    cached_text_model, _, _ = server.get_cached_model("fake-text")

    assert cached_text_model is text_model
    assert server.runtime.response_generator is text_generator


def _drain(handle, timeout=2.0):
    deadline = time.time() + timeout
    chunks = []
    while time.time() < deadline:
        chunk = handle.result_queue.get(timeout=timeout)
        if chunk.kind == "data":
            chunks.append(chunk.payload)
        elif chunk.kind == "error":
            raise chunk.error
        elif chunk.kind == "done":
            return chunks
    raise TimeoutError("timed out waiting for audio queue results")
