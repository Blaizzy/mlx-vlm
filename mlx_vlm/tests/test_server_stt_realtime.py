import base64
import threading
import time
from types import SimpleNamespace

import mlx_vlm.server as server
import pytest
from fastapi.testclient import TestClient
from mlx_vlm.server import audio
from mlx_vlm.tests.test_server_audio import reset_audio_runtime


class Session:
    input_sample_rate = 16000

    def __init__(self):
        self.done = False
        self.pending = False
        self.cancelled = threading.Event()
        self.threads = []

    def feed(self, samples):
        self.threads.append(threading.get_ident())
        self.pending = True

    def step(self, **kwargs):
        self.threads.append(threading.get_ident())
        if self.pending:
            self.pending = False
            return ["hello"]
        return []

    def close(self):
        self.threads.append(threading.get_ident())
        self.done = True

    def cancel(self):
        self.threads.append(threading.get_ident())
        self.cancelled.set()


@pytest.fixture
def session(monkeypatch, reset_audio_runtime):
    result = Session()

    def load(*args, **kwargs):
        assert kwargs == {"model_kind": "audio_stt"}
        result.threads.append(threading.get_ident())
        return SimpleNamespace(create_streaming_session=lambda **kw: result), None, None

    monkeypatch.setattr(audio, "get_cached_model", load)
    return result


def receive_until(ws, kind):
    while True:
        message = ws.receive_json()
        assert message["type"] != "error", message
        if message["type"] == kind:
            return message


def test_live_partial_final_and_worker_ownership(session):
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\0\0" * 320).decode(),
                }
            )
            assert (
                receive_until(ws, "conversation.item.input_audio_transcription.delta")[
                    "delta"
                ]
                == "hello"
            )
            assert not session.done
            ws.send_json({"type": "input_audio_buffer.commit"})
            assert (
                receive_until(
                    ws, "conversation.item.input_audio_transcription.completed"
                )["transcript"]
                == "hello"
            )
    assert session.cancelled.wait(2)
    assert len(set(session.threads)) == 1
    assert session.threads[0] != threading.get_ident()


@pytest.mark.parametrize(
    "message",
    [
        {"type": "input_audio_buffer.append", "audio": "!"},
        {"type": "input_audio_buffer.append", "audio": "AA=="},
        {"type": "input_audio_buffer.append", "audio": 1},
        {"type": "session.update", "session": {"input_audio_sample_rate": 48000}},
        {
            "type": "session.update",
            "session": {"turn_detection": {"type": "server_vad"}},
        },
        {"type": "unknown"},
        [],
    ],
)
def test_invalid_input_is_cancelled(session, message):
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            ws.receive_json()
            ws.send_json(message)
            assert ws.receive_json()["type"] == "error"
    assert session.cancelled.wait(2)


def test_contention_and_disconnect(session):
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            ws.receive_json()
            with client.websocket_connect(
                "/v1/audio/transcriptions/realtime?model=fake"
            ) as other:
                assert "busy" in other.receive_json()["error"]["message"]
            with pytest.raises(Exception, match="busy"):
                audio._get_audio_queue().submit(
                    kind="stt", model_name="fake", payload={}
                )
    assert session.cancelled.wait(2)


def test_oversized_message(session):
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            ws.receive_json()
            ws.send_text(" " * (128 * 1024 + 1))
            assert "size limit" in ws.receive_json()["error"]["message"]
    assert session.cancelled.wait(2)


def test_post_commit_audio_is_rejected(session):
    session.close = lambda: None
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.commit"})
            receive_until(ws, "input_audio_buffer.committed")
            ws.send_json({"type": "input_audio_buffer.append", "audio": "AAA="})
            assert "already been committed" in ws.receive_json()["error"]["message"]
    assert not session.pending


def test_shutdown_resolves_active_and_pending(monkeypatch, reset_audio_runtime):
    started = threading.Event()

    def run(request):
        request.mark_ready()
        started.set()
        assert request.cancel_event.wait(2)

    monkeypatch.setattr(audio, "_run_stt_request", run)
    worker = audio.AudioRequestQueue()
    active = worker.submit(kind="stt", model_name="fake", payload={})
    assert started.wait(2)
    pending = worker.submit(kind="stt", model_name="fake", payload={})
    worker.stop_and_join()
    assert active.cancel_event.is_set()
    assert pending.ready_event.is_set()
    assert pending.ready_error is not None
    assert pending.result_queue.get_nowait().kind == "done"
    with pytest.raises(Exception, match="stopping"):
        worker.submit(kind="stt", model_name="fake", payload={})


def test_authentication_applies_to_websocket(session, monkeypatch):
    from starlette.testclient import WebSocketDenialResponse

    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "synthetic-test-key")
    with TestClient(server.app) as client:
        with pytest.raises(WebSocketDenialResponse) as denied:
            with client.websocket_connect(
                "/v1/audio/transcriptions/realtime?model=fake"
            ):
                pass
        assert denied.value.status_code == 401
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake",
            headers={"Authorization": "Bearer synthetic-test-key"},
        ) as ws:
            assert ws.receive_json()["type"] == "session.created"


def test_cold_step_backpressure_does_not_drop_audio(session):
    original_step = session.step
    first = True

    def slow_step(**kwargs):
        nonlocal first
        if first:
            first = False
            time.sleep(0.3)
        return original_step(**kwargs)

    session.step = slow_step
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            ws.receive_json()
            for _ in range(80):
                ws.send_json({"type": "input_audio_buffer.append", "audio": "AAA="})
            ws.send_json({"type": "input_audio_buffer.commit"})
            result = receive_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )
            assert result["transcript"] == "hello" * 80


def test_incompatible_session_is_rejected(session, monkeypatch):
    monkeypatch.setattr(
        audio,
        "get_cached_model",
        lambda *a, **kw: (
            SimpleNamespace(create_streaming_session=lambda **kw: object()),
            None,
            None,
        ),
    )
    with TestClient(server.app) as client:
        with client.websocket_connect(
            "/v1/audio/transcriptions/realtime?model=fake"
        ) as ws:
            assert "incompatible" in ws.receive_json()["error"]["message"]
