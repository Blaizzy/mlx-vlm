import base64
import os

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm.models.nemotron_voicechat.streaming import VoiceChatEvent
from mlx_vlm.server import realtime


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
                kind="function_delta",
                frame_index=0,
                token_id=43,
                delta="{",
                text="{",
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (30, 30.0)],
)
def test_realtime_websocket_accepts_explicit_session_limit(
    realtime_client, value, expected
):
    client, loaded, _ = realtime_client
    with client.websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "fake-voicechat",
                    "max_streaming_seconds": value,
                },
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


def test_realtime_websocket_cancels_and_releases_session(realtime_client):
    client, loaded, engine = realtime_client
    with client.websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {"type": "session.update", "session": {"model": "fake-voicechat"}}
        )
        websocket.receive_json()
        websocket.send_json({"type": "response.cancel"})
        assert websocket.receive_json()["type"] == "response.cancelled"

    assert loaded.session.closed is True
    assert engine.try_reserve("next-session") is True
    engine.release("next-session")


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
