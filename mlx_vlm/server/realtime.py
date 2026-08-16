"""Model-specific online VoiceChat WebSocket transport."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import mlx.core as mx
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .runtime import runtime

logger = logging.getLogger("mlx_vlm.server")


def load_realtime_voicechat(model_name: str):
    from ..utils import load

    model, processor = load(
        model_name,
        lazy=True,
        strict=True,
        trust_remote_code=False,
    )
    if not hasattr(model, "create_session"):
        raise TypeError(f"{model_name!r} is not a realtime VoiceChat model")
    return model.create_session(processor)


@dataclass
class _Command:
    kind: Literal["open", "push", "flush", "cancel", "close"]
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    result: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))


class RealtimeVoiceChatEngine:
    """One long-lived MLX worker with a hard single-session reservation."""

    def __init__(self, loader=None):
        self.loader = loader or load_realtime_voicechat
        self._commands: queue.Queue[_Command | None] = queue.Queue()
        self._reservation_lock = threading.Lock()
        self._reserved_session_id: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mlx-vlm-realtime",
            daemon=True,
        )
        self._thread.start()

    def is_worker_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def try_reserve(self, session_id: str) -> bool:
        with self._reservation_lock:
            if self._reserved_session_id is not None:
                return False
            self._reserved_session_id = session_id
            return True

    def release(self, session_id: str) -> None:
        with self._reservation_lock:
            if self._reserved_session_id == session_id:
                self._reserved_session_id = None

    def qsize(self) -> int:
        return self._commands.qsize()

    def _call(self, kind: str, session_id: str, **payload):
        command = _Command(kind=kind, session_id=session_id, payload=payload)
        self._commands.put(command)
        success, value = command.result.get()
        if success:
            return value
        raise value

    def open(self, session_id: str, **config):
        return self._call("open", session_id, **config)

    def push(self, session_id: str, samples: np.ndarray, sample_rate: int):
        return self._call(
            "push",
            session_id,
            samples=samples,
            sample_rate=sample_rate,
        )

    def flush(self, session_id: str, pad_partial: bool = True):
        return self._call("flush", session_id, pad_partial=pad_partial)

    def cancel(self, session_id: str):
        return self._call("cancel", session_id)

    def close(self, session_id: str):
        return self._call("close", session_id)

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._commands.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        loaded_name: str | None = None
        loaded_model = None
        session_id: str | None = None
        session = None

        while not self._stop.is_set():
            command = self._commands.get()
            if command is None:
                break
            try:
                if command.kind == "open":
                    if session is not None:
                        raise RuntimeError("a realtime session is already active")
                    model_name = command.payload["model"]
                    if loaded_model is None or loaded_name != model_name:
                        loaded_model = self.loader(model_name)
                        loaded_name = model_name
                    session = loaded_model.create_streaming_session(
                        system_prompt=command.payload.get("system_prompt"),
                        seed=command.payload.get("seed", 0),
                        max_streaming_seconds=command.payload.get(
                            "max_streaming_seconds"
                        ),
                    )
                    session_id = command.session_id
                    value = {
                        "input_sample_rate": session.input_sample_rate,
                        "output_sample_rate": session.output_sample_rate,
                        "frame_samples": session.frame_samples,
                    }
                else:
                    if session is None or session_id != command.session_id:
                        raise RuntimeError("realtime session is not active")
                    if command.kind == "push":
                        events = session.push_audio(
                            command.payload["samples"],
                            sample_rate=command.payload["sample_rate"],
                        )
                        value = [_serialize_model_event(event) for event in events]
                    elif command.kind == "flush":
                        events = session.flush(
                            pad_partial=command.payload["pad_partial"]
                        )
                        value = [_serialize_model_event(event) for event in events]
                    elif command.kind == "cancel":
                        events = session.cancel()
                        value = [_serialize_model_event(event) for event in events]
                    elif command.kind == "close":
                        if not session.closed:
                            session.cancel()
                        session = None
                        session_id = None
                        value = None
                    else:
                        raise ValueError(f"unsupported realtime command {command.kind}")
                command.result.put((True, value))
            except Exception as exc:
                logger.exception("Realtime VoiceChat command failed")
                command.result.put((False, exc))
            finally:
                # Keep MLX's reusable buffers hot between native audio frames.
                # Clearing on every push materially increases online latency.
                if command.kind == "close":
                    mx.clear_cache()


_ENGINE_LOCK = threading.Lock()


def get_realtime_engine() -> RealtimeVoiceChatEngine:
    with _ENGINE_LOCK:
        if runtime.realtime_engine is None:
            runtime.realtime_engine = RealtimeVoiceChatEngine()
        return runtime.realtime_engine


def _pcm16_from_base64(value: str) -> np.ndarray:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio must be valid base64 PCM16") from exc
    if len(raw) % 2:
        raise ValueError("PCM16 audio must contain an even number of bytes")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _audio_to_base64(samples: mx.array) -> str:
    values = np.asarray(samples, dtype=np.float32)
    pcm = np.round(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _serialize_model_event(event) -> dict[str, Any]:
    common = {"frame_index": event.frame_index}
    if event.kind == "assistant_text_delta":
        return {
            "type": "response.text.delta",
            **common,
            "token_id": event.token_id,
            "delta": event.delta,
            "text": event.text,
        }
    if event.kind == "function_delta":
        return {
            "type": "response.function.delta",
            **common,
            "token_id": event.token_id,
            "delta": event.delta,
            "text": event.text,
        }
    if event.kind == "user_transcript_delta":
        return {
            "type": "conversation.item.input_audio_transcription.delta",
            **common,
            "delta": event.delta,
            "transcript": event.text,
        }
    if event.kind == "audio":
        return {
            "type": "response.audio.delta",
            **common,
            "delta": _audio_to_base64(event.samples),
            "format": "pcm16",
            "sample_rate": event.sample_rate,
            "channels": 1,
            "audio_codes": event.audio_codes.tolist(),
        }
    if event.kind == "done":
        return {"type": "response.done", **common}
    if event.kind == "cancelled":
        return {"type": "response.cancelled", **common}
    raise ValueError(f"unsupported VoiceChat event kind {event.kind}")


async def realtime_voicechat_endpoint(websocket: WebSocket):
    engine = get_realtime_engine()
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    await websocket.accept()

    async def send(payload: dict[str, Any]) -> None:
        await websocket.send_json(
            {"event_id": f"event_{uuid.uuid4().hex[:16]}", **payload}
        )

    async def send_error(message: str, *, code: str = "invalid_request") -> None:
        await send({"type": "error", "error": {"code": code, "message": message}})

    if not engine.try_reserve(session_id):
        await send_error(
            "another realtime VoiceChat session is already active",
            code="server_busy",
        )
        await websocket.close(code=1013)
        return

    configured = False
    frame_samples = 1280
    await send(
        {
            "type": "session.created",
            "session": {
                "id": session_id,
                "state": "configuring",
                "input_audio_format": {"type": "pcm16", "sample_rate": 16000},
                "output_audio_format": {"type": "pcm16", "sample_rate": 22050},
            },
        }
    )

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except ValueError:
                await send_error("message must be a JSON object")
                continue
            message_type = message.get("type", "")

            if message_type == "session.update":
                if configured:
                    await send_error("the active session cannot be reconfigured")
                    continue
                payload = message.get("session") or {}
                model_name = payload.get("model") or websocket.query_params.get("model")
                if not model_name:
                    await send_error("session.update.session.model is required")
                    continue
                try:
                    max_streaming_seconds = payload.get("max_streaming_seconds")
                    if max_streaming_seconds is not None:
                        max_streaming_seconds = float(max_streaming_seconds)
                    details = await asyncio.to_thread(
                        engine.open,
                        session_id,
                        model=model_name,
                        system_prompt=payload.get("system_prompt"),
                        seed=int(payload.get("seed", 0)),
                        max_streaming_seconds=max_streaming_seconds,
                    )
                except Exception as exc:
                    await send_error(str(exc), code="session_initialization_failed")
                    continue
                configured = True
                frame_samples = details["frame_samples"]
                await send(
                    {
                        "type": "session.updated",
                        "session": {
                            "id": session_id,
                            "state": "ready",
                            "model": model_name,
                            "frame_samples": details["frame_samples"],
                            "input_audio_format": {
                                "type": "pcm16",
                                "sample_rate": details["input_sample_rate"],
                            },
                            "output_audio_format": {
                                "type": "pcm16",
                                "sample_rate": details["output_sample_rate"],
                            },
                        },
                    }
                )
                continue

            if message_type == "session.ping":
                await send({"type": "session.pong"})
                continue

            if not configured:
                await send_error("send session.update before audio")
                continue

            try:
                if message_type == "input_audio_buffer.append":
                    audio = _pcm16_from_base64(message.get("audio", ""))
                    sample_rate = int(message.get("sample_rate", 16000))
                    # Bound each worker call to one native frame so a client can
                    # submit a large network chunk without delaying every output
                    # event until the whole chunk has finished inference.
                    for start in range(0, max(audio.shape[0], 1), frame_samples):
                        events = await asyncio.to_thread(
                            engine.push,
                            session_id,
                            audio[start : start + frame_samples],
                            sample_rate,
                        )
                        for event in events:
                            await send(event)
                elif message_type == "input_audio_buffer.commit":
                    await send({"type": "input_audio_buffer.committed"})
                    events = await asyncio.to_thread(
                        engine.flush,
                        session_id,
                        bool(message.get("pad_partial", True)),
                    )
                    for event in events:
                        await send(event)
                    break
                elif message_type in {"session.cancel", "response.cancel"}:
                    events = await asyncio.to_thread(engine.cancel, session_id)
                    for event in events:
                        await send(event)
                    break
                else:
                    await send_error(f"unsupported event type {message_type!r}")
            except Exception as exc:
                await send_error(str(exc), code="inference_error")
    except WebSocketDisconnect:
        pass
    finally:
        if configured:
            try:
                await asyncio.to_thread(engine.close, session_id)
            except Exception:
                logger.exception("Failed to close realtime VoiceChat session")
        engine.release(session_id)


def register_routes(app, deps=None):
    del deps
    app.websocket("/v1/realtime")(realtime_voicechat_endpoint)
