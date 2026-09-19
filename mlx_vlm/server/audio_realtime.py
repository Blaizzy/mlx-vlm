"""Bounded, single-utterance live STT on the existing audio worker/cache."""

import asyncio
import base64
import json
import queue
import time

import anyio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

MAX_MESSAGE_BYTES = 128 * 1024
IDLE_TIMEOUT = 30
SESSION_TIMEOUT = 120


def run_request(request, get_cached_model):
    session = None
    try:
        model, _, _ = get_cached_model(request.model_name, model_kind="audio_stt")
        factory = getattr(model, "create_streaming_session", None)
        if not callable(factory):
            raise ValueError("Model does not support live-input STT")
        session = factory(temperature=0)
        if not all(
            callable(getattr(session, name, None)) for name in ("feed", "step", "close")
        ) or not all(hasattr(session, name) for name in ("done", "input_sample_rate")):
            raise ValueError("Model has an incompatible live-input session")
        request.mark_ready()
        request.emit_data(
            {
                "type": "session.created",
                "session": {
                    "id": request.request_id,
                    "model": request.model_name,
                    "input_audio_format": "pcm16",
                    "input_audio_sample_rate": session.input_sample_rate,
                    "turn_detection": None,
                    "max_duration_seconds": SESSION_TIMEOUT,
                },
            }
        )
        committed = False
        started = False
        transcript = ""
        deadline = time.monotonic() + SESSION_TIMEOUT
        samples_received = 0
        while not request.cancel_event.is_set():
            if time.monotonic() >= deadline:
                raise ValueError("Session duration limit exceeded")
            try:
                message = request.payload.get(timeout=0.01)
            except queue.Empty:
                message = None
            if message is not None:
                kind = message.get("type")
                if kind == "session.cancel":
                    return
                if committed:
                    raise ValueError("Audio has already been committed")
                if kind == "session.update":
                    config = message.get("session", {})
                    if not isinstance(config, dict) or any(
                        config.get(key, value) != value
                        for key, value in {
                            "model": request.model_name,
                            "input_audio_format": "pcm16",
                            "input_audio_sample_rate": session.input_sample_rate,
                            "turn_detection": None,
                        }.items()
                    ):
                        raise ValueError(
                            "Only native-rate mono PCM16 and manual commit are supported"
                        )
                    if set(config) - {
                        "model",
                        "input_audio_format",
                        "input_audio_sample_rate",
                        "turn_detection",
                    }:
                        raise ValueError("Unsupported session setting")
                    request.emit_data({"type": "session.updated", "session": config})
                elif kind == "input_audio_buffer.append":
                    encoded = message.get("audio")
                    if not isinstance(encoded, str):
                        raise ValueError("audio must be base64 PCM16")
                    raw = base64.b64decode(encoded, validate=True)
                    if not raw or len(raw) % 2:
                        raise ValueError("audio must contain complete PCM16 samples")
                    samples_received += len(raw) // 2
                    if samples_received > SESSION_TIMEOUT * session.input_sample_rate:
                        raise ValueError("Audio duration limit exceeded")
                    session.feed(
                        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                    )
                    if not started:
                        started = True
                        request.emit_data(
                            {
                                "type": "conversation.item.added",
                                "item_id": request.request_id,
                            }
                        )
                elif kind == "input_audio_buffer.commit":
                    session.close()
                    committed = True
                    request.emit_data(
                        {
                            "type": "input_audio_buffer.committed",
                            "item_id": request.request_id,
                        }
                    )
                else:
                    raise ValueError("Unsupported event type")
            for delta in session.step(max_decode_tokens=8):
                transcript += delta
                request.emit_data(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": request.request_id,
                        "delta": delta,
                    }
                )
            if session.done:
                request.emit_data(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": request.request_id,
                        "transcript": transcript,
                    }
                )
                return
    finally:
        if session is not None and callable(getattr(session, "cancel", None)):
            session.cancel()
        while True:
            try:
                request.payload.get_nowait()
            except queue.Empty:
                break


async def audio_realtime_endpoint(websocket: WebSocket, model: str):
    from .audio import _get_audio_queue

    await websocket.accept()
    handle = None
    tasks = []
    try:
        incoming = queue.Queue(maxsize=64)
        handle = _get_audio_queue().submit(
            kind="stt_realtime", model_name=model, payload=incoming
        )

        async def receive():
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), IDLE_TIMEOUT)
                if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    raise ValueError("Message size limit exceeded")
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("Expected a JSON object")
                # Apply transport backpressure while the first encoder step
                # compiles; do not drop valid paced audio during a cold start.
                deadline = time.monotonic() + IDLE_TIMEOUT
                while True:
                    try:
                        incoming.put_nowait(message)
                        break
                    except queue.Full:
                        if time.monotonic() >= deadline:
                            raise ValueError("Audio input queue stalled") from None
                        await asyncio.sleep(0.01)

        async def transmit():
            while not handle.cancel_event.is_set():
                try:
                    chunk = handle.result_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if chunk.kind == "error":
                    raise chunk.error
                if chunk.kind == "done":
                    return
                await asyncio.wait_for(websocket.send_json(chunk.payload), 5)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(transmit())]
        done, _ = await asyncio.wait(
            tasks, timeout=SESSION_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            raise ValueError("Session duration limit exceeded")
        for task in done:
            task.result()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        try:
            await asyncio.wait_for(
                websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "message": str(getattr(exc, "detail", exc))
                            or type(exc).__name__
                        },
                    }
                ),
                5,
            )
        except Exception:
            pass
    finally:
        if handle is not None:
            handle.cancel()
        for task in tasks:
            task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass
