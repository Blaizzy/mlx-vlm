"""HTTP audio endpoints and realtime voice sessions."""

import base64
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.audio as server_audio
from mlx_vlm.models.nemotron_voicechat.streaming import VoiceChatEvent
from mlx_vlm.server import realtime

# HTTP audio endpoints


@pytest.fixture
def client(reset_audio_runtime):
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
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


@pytest.mark.usefixtures("reset_audio_runtime")
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


@pytest.mark.usefixtures("reset_audio_runtime")
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


@pytest.mark.usefixtures("reset_audio_runtime")
@pytest.mark.parametrize(
    "endpoint, options, text, forwarded",
    [
        (
            "transcriptions",
            {"prompt": "prior context", "language": "en"},
            "This is a test transcription.",
            {"context": "prior context", "language": "en"},
        ),
        ("transcriptions", {"response_format": "text"}, "Plain text transcript.", {}),
        ("translations", {}, "Translated transcript.", {"task": "translate"}),
    ],
    ids=["transcription-json", "transcription-text", "translation"],
)
def test_audio_transcription_request(
    client, monkeypatch, endpoint, options, text, forwarded
):
    fake = FakeSTTModel({"text": text})
    loader = Mock(return_value=(fake, None, SimpleNamespace(model_type="audio")))
    monkeypatch.setattr(server, "get_cached_model", loader)
    monkeypatch.setattr(
        server_audio,
        "audio_read",
        lambda buffer, always_2d=False: (np.zeros(160, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)
    response = client.post(
        "/v1/audio/" + endpoint,
        files={"file": ("test.wav", b"audio-bytes", "audio/wav")},
        data={"model": "fake-stt", **options},
    )
    assert response.status_code == 200
    if options.get("response_format") == "text":
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == text
    else:
        assert response.json() == {"text": text}
    assert {key: fake.calls[0][key] for key in forwarded} == forwarded
    loader.assert_called_once_with("fake-stt", model_kind="audio_stt")


@pytest.mark.usefixtures("reset_audio_runtime")
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


@pytest.mark.usefixtures("reset_audio_runtime")
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


# Realtime voice sessions


class _FakeStreamingSession:
    input_sample_rate = 16000
    output_sample_rate = 22050
    frame_samples = 1280

    def __init__(self):
        self.closed = False
        self.push_threads = []

    def push_audio(self, samples, sample_rate):
        import threading

        self.push_threads.append(threading.current_thread().name)
        assert sample_rate == 16000
        assert samples.shape == (1280,)
        return [
            VoiceChatEvent(
                kind="assistant_text_delta",
                frame_index=0,
                token_id=42,
                delta="hello",
                text="hello",
            ),
            VoiceChatEvent(
                kind="function_delta", frame_index=0, token_id=43, delta="{", text="{"
            ),
            VoiceChatEvent(
                kind="audio",
                frame_index=0,
                samples=mx.zeros((1764,)),
                sample_rate=22050,
                audio_codes=mx.zeros((31,), dtype=mx.int32),
            ),
        ]

    def flush(self, pad_partial=True):
        self.closed = True
        return [VoiceChatEvent(kind="done", frame_index=1)]

    def cancel(self):
        self.closed = True
        return [VoiceChatEvent(kind="cancelled", frame_index=0)]


class _FakeLoadedModel:
    def __init__(self):
        self.session = None
        self.configs = []

    def create_streaming_session(self, **kwargs):
        self.configs.append(kwargs)
        self.session = _FakeStreamingSession()
        return self.session


def test_realtime_loader_uses_generic_load_and_model_session(monkeypatch):
    calls = []
    created = object()
    processor = object()

    class Model:
        def create_session(self, value):
            calls.append(("create_session", value))
            return created

    def fake_load(model_name, **kwargs):
        calls.append(("load", model_name, kwargs))
        return Model(), processor

    monkeypatch.setattr("mlx_vlm.utils.load", fake_load)

    assert realtime.load_realtime_voicechat("mlx-community/model") is created
    assert calls == [
        (
            "load",
            "mlx-community/model",
            {"lazy": True, "strict": True, "trust_remote_code": False},
        ),
        ("create_session", processor),
    ]


@pytest.fixture
def realtime_client(monkeypatch):
    os.environ.pop("MLX_VLM_SERVER_API_KEY", None)
    if server.runtime.realtime_engine is not None:
        server.runtime.realtime_engine.stop_and_join()
    loaded = _FakeLoadedModel()
    engine = realtime.RealtimeVoiceChatEngine(loader=lambda _: loaded)
    monkeypatch.setattr(server.runtime, "realtime_engine", engine)
    with TestClient(server.app) as client:
        yield client, loaded, engine
    if server.runtime.realtime_engine is not None:
        server.runtime.realtime_engine.stop_and_join()
        server.runtime.realtime_engine = None


def test_realtime_websocket_streams_json_events(realtime_client):
    client, loaded, _ = realtime_client
    pcm = np.zeros(1280, dtype="<i2").tobytes()

    with client.websocket_connect("/v1/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "fake-voicechat",
                    "system_prompt": "Be brief.",
                    "seed": 7,
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["state"] == "ready"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
                "sample_rate": 16000,
            }
        )
        text = websocket.receive_json()
        function = websocket.receive_json()
        audio = websocket.receive_json()
        assert text["type"] == "response.text.delta"
        assert text["delta"] == "hello"
        assert function["type"] == "response.function.delta"
        assert function["delta"] == "{"
        assert audio["type"] == "response.audio.delta"
        assert len(base64.b64decode(audio["delta"])) == 1764 * 2

        websocket.send_json({"type": "input_audio_buffer.commit"})
        assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
        assert websocket.receive_json()["type"] == "response.done"

    assert loaded.configs[0]["system_prompt"] == "Be brief."
    assert loaded.configs[0]["seed"] == 7
    assert loaded.configs[0]["max_streaming_seconds"] is None
    assert loaded.session.push_threads == ["mlx-vlm-realtime"]


@pytest.mark.parametrize(("value", "expected"), [(None, None), (30, 30.0)])
def test_realtime_websocket_accepts_explicit_session_limit(
    realtime_client, value, expected
):
    client, loaded, _ = realtime_client
    with client.websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.update",
                "session": {"model": "fake-voicechat", "max_streaming_seconds": value},
            }
        )
        assert websocket.receive_json()["type"] == "session.updated"
        websocket.send_json({"type": "response.cancel"})
        assert websocket.receive_json()["type"] == "response.cancelled"

    assert loaded.configs[0]["max_streaming_seconds"] == expected


def test_realtime_websocket_rejects_second_active_session(realtime_client):
    client, _, _ = realtime_client
    with client.websocket_connect("/v1/realtime") as first:
        assert first.receive_json()["type"] == "session.created"
        with client.websocket_connect("/v1/realtime") as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "server_busy"


def test_realtime_websocket_requires_native_pcm_rate(realtime_client):
    client, _, _ = realtime_client
    pcm = np.zeros(1280, dtype="<i2").tobytes()
    with client.websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {"type": "session.update", "session": {"model": "fake-voicechat"}}
        )
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
                "sample_rate": 24000,
            }
        )
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "inference_error"


def test_streaming_session_buffers_arbitrary_chunk_boundaries():
    from mlx_vlm.models.nemotron_voicechat.streaming import VoiceChatStreamingSession

    stream = VoiceChatStreamingSession.__new__(VoiceChatStreamingSession)
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
