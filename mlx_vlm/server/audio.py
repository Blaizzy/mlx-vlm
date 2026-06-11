import asyncio
import inspect
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from pydantic import Field

from .runtime import runtime
from .schemas import FlexibleBaseModel

logger = logging.getLogger("mlx_vlm.server")

get_cached_model = None
_build_metrics_envelope = None


class AudioSpeechRequest(FlexibleBaseModel):
    model: str
    input: str
    instruct: Optional[str] = None
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    gender: Optional[str] = "male"
    pitch: Optional[float] = 1.0
    lang_code: Optional[str] = "a"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.0
    response_format: str = "mp3"
    stream: bool = False
    streaming_interval: float = 2.0
    max_tokens: int = 1200
    verbose: bool = False


class AudioTranscriptionRequest(FlexibleBaseModel):
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    response_format: str = "json"
    verbose: bool = False
    max_tokens: int = 1024
    chunk_duration: float = 30.0
    frame_threshold: int = 25
    stream: bool = False
    context: Optional[str] = None
    prefill_step_size: int = 2048
    text: Optional[str] = None
    word_timestamps: bool = False
    timestamp_granularities: Optional[List[str]] = Field(default=None)


@dataclass
class SpeechTaskPayload:
    request: AudioSpeechRequest


@dataclass
class TranscriptionTaskPayload:
    request: AudioTranscriptionRequest
    filename: str
    audio: np.ndarray
    sample_rate: int
    translate: bool = False


@dataclass
class AudioResultChunk:
    kind: Literal["data", "error", "done"]
    payload: Any = None
    error: BaseException | None = None


@dataclass
class AudioInferenceRequest:
    kind: Literal["tts", "stt"]
    model_name: str
    payload: Any
    request_id: str = ""
    result_queue: "queue.Queue[AudioResultChunk]" = None
    cancel_event: threading.Event = None
    ready_event: threading.Event = None
    ready_error: BaseException | None = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = uuid.uuid4().hex
        if self.result_queue is None:
            self.result_queue = queue.Queue()
        if self.cancel_event is None:
            self.cancel_event = threading.Event()
        if self.ready_event is None:
            self.ready_event = threading.Event()

    def emit_data(self, payload: Any) -> None:
        self.result_queue.put(AudioResultChunk(kind="data", payload=payload))

    def emit_error(self, error: BaseException) -> None:
        self.result_queue.put(AudioResultChunk(kind="error", error=error))

    def emit_done(self) -> None:
        self.result_queue.put(AudioResultChunk(kind="done"))

    def mark_ready(self, error: BaseException | None = None) -> None:
        self.ready_error = error
        self.ready_event.set()


@dataclass
class AudioInferenceHandle:
    request_id: str
    result_queue: "queue.Queue[AudioResultChunk]"
    cancel_event: threading.Event
    ready_event: threading.Event
    request: AudioInferenceRequest

    @property
    def ready_error(self) -> BaseException | None:
        return self.request.ready_error

    def cancel(self) -> None:
        self.cancel_event.set()


class AudioRequestQueue:
    def __init__(self):
        self._requests: "queue.Queue[AudioInferenceRequest | None]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def qsize(self) -> int:
        return self._requests.qsize()

    def is_worker_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def submit(
        self,
        *,
        kind: Literal["tts", "stt"],
        model_name: str,
        payload: Any,
    ) -> AudioInferenceHandle:
        request = AudioInferenceRequest(
            kind=kind,
            model_name=model_name,
            payload=payload,
        )
        self._requests.put(request)
        return AudioInferenceHandle(
            request_id=request.request_id,
            result_queue=request.result_queue,
            cancel_event=request.cancel_event,
            ready_event=request.ready_event,
            request=request,
        )

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._requests.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            request = self._requests.get()
            if request is None:
                break
            if request.cancel_event.is_set():
                request.mark_ready()
                request.emit_done()
                continue
            try:
                if request.kind == "tts":
                    _run_tts_request(request)
                elif request.kind == "stt":
                    _run_stt_request(request)
                else:
                    raise ValueError(f"Unsupported audio request kind: {request.kind}")
            except Exception as exc:
                traceback.print_exc()
                if not request.ready_event.is_set():
                    request.mark_ready(exc)
                request.emit_error(exc)
            finally:
                request.emit_done()
                mx.clear_cache()


_AUDIO_QUEUE_LOCK = threading.Lock()


def _get_audio_queue() -> AudioRequestQueue:
    with _AUDIO_QUEUE_LOCK:
        if runtime.audio_queue is None:
            runtime.audio_queue = AudioRequestQueue()
        return runtime.audio_queue


def register_routes(app, deps):
    global get_cached_model, _build_metrics_envelope

    get_cached_model = deps.get_cached_model
    _build_metrics_envelope = deps.build_metrics_envelope

    app.post("/audio/speech", response_model=None)(audio_speech_endpoint)
    app.post("/v1/audio/speech", response_model=None, include_in_schema=False)(
        audio_speech_endpoint
    )
    app.post("/audio/transcriptions", response_model=None)(
        audio_transcriptions_endpoint
    )
    app.post(
        "/v1/audio/transcriptions",
        response_model=None,
        include_in_schema=False,
    )(audio_transcriptions_endpoint)
    app.post("/audio/translations", response_model=None)(audio_translations_endpoint)
    app.post(
        "/v1/audio/translations",
        response_model=None,
        include_in_schema=False,
    )(audio_translations_endpoint)


async def audio_speech_endpoint(payload: AudioSpeechRequest, request: Request):
    endpoint = "/v1/audio/speech"
    request_start = time.perf_counter()
    runtime.metrics.begin_request(
        endpoint=endpoint, model=payload.model, stream=payload.stream
    )
    try:
        if not payload.input:
            raise HTTPException(status_code=400, detail="Missing input.")
        if payload.ref_audio and isinstance(payload.ref_audio, str):
            if not os.path.exists(payload.ref_audio):
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference audio file not found: {payload.ref_audio}",
                )

        handle = _get_audio_queue().submit(
            kind="tts",
            model_name=payload.model,
            payload=SpeechTaskPayload(request=payload),
        )
        media_type = f"audio/{payload.response_format}"
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        }

        if payload.stream:
            await _wait_until_ready(handle)
            return StreamingResponse(
                _stream_audio_handle(
                    handle,
                    request,
                    endpoint=endpoint,
                    model=payload.model,
                    request_start=request_start,
                    stream=True,
                    media_kind="speech",
                ),
                media_type=media_type,
                headers=headers,
            )

        chunks = await _collect_audio_handle(handle)
        body = b"".join(_ensure_bytes(chunk) for chunk in chunks)
        _record_audio_success(
            endpoint=endpoint,
            model=payload.model,
            stream=False,
            request_start=request_start,
            audio_count=0,
        )
        return Response(content=body, media_type=media_type, headers=headers)
    except HTTPException as exc:
        _record_audio_failure(endpoint, payload.model, payload.stream, exc)
        raise
    except Exception as exc:
        _record_audio_failure(endpoint, payload.model, payload.stream, exc)
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {exc}")


async def audio_transcriptions_endpoint(request: Request):
    return await _audio_transcription_response(request, translate=False)


async def audio_translations_endpoint(request: Request):
    return await _audio_transcription_response(request, translate=True)


async def _audio_transcription_response(request: Request, *, translate: bool):
    endpoint = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
    request_start = time.perf_counter()
    payload = await _parse_transcription_request(request, translate=translate)
    stream_response = (
        payload.request.response_format == "ndjson" or payload.request.stream
    )
    runtime.metrics.begin_request(
        endpoint=endpoint,
        model=payload.request.model,
        stream=stream_response,
    )
    try:
        handle = _get_audio_queue().submit(
            kind="stt",
            model_name=payload.request.model,
            payload=payload,
        )
        if stream_response:
            await _wait_until_ready(handle)
            return StreamingResponse(
                _stream_audio_handle(
                    handle,
                    request,
                    endpoint=endpoint,
                    model=payload.request.model,
                    request_start=request_start,
                    stream=True,
                    media_kind="transcription",
                ),
                media_type="application/x-ndjson",
            )

        chunks = await _collect_audio_handle(handle)
        result = _transcription_result_from_chunks(chunks)
        _record_audio_success(
            endpoint=endpoint,
            model=payload.request.model,
            stream=False,
            request_start=request_start,
            audio_count=1,
        )
        return _format_transcription_response(result, payload.request.response_format)
    except HTTPException as exc:
        _record_audio_failure(endpoint, payload.request.model, stream_response, exc)
        raise
    except Exception as exc:
        _record_audio_failure(endpoint, payload.request.model, stream_response, exc)
        raise HTTPException(
            status_code=500, detail=f"Audio transcription failed: {exc}"
        )


async def _parse_transcription_request(
    request: Request,
    *,
    translate: bool,
) -> TranscriptionTaskPayload:
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid multipart form: {exc}"
        ) from exc

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="Missing file.")

    model = _clean_form_value(form.get("model"))
    if not model:
        raise HTTPException(status_code=400, detail="Missing model.")

    timestamp_granularities = _form_list(form, "timestamp_granularities")
    if not timestamp_granularities:
        timestamp_granularities = _form_list(form, "timestamp_granularities[]")

    payload = AudioTranscriptionRequest(
        model=model,
        language=_clean_form_value(form.get("language")),
        prompt=_clean_form_value(form.get("prompt")),
        temperature=_form_float(form.get("temperature")),
        response_format=_clean_form_value(form.get("response_format")) or "json",
        verbose=_form_bool(form.get("verbose"), default=False),
        max_tokens=_form_int(form.get("max_tokens"), default=1024),
        chunk_duration=_form_float(form.get("chunk_duration"), default=30.0),
        frame_threshold=_form_int(form.get("frame_threshold"), default=25),
        stream=_form_bool(form.get("stream"), default=False),
        context=_clean_form_value(form.get("context")),
        prefill_step_size=_form_int(form.get("prefill_step_size"), default=2048),
        text=_clean_form_value(form.get("text")),
        word_timestamps=_form_bool(form.get("word_timestamps"), default=False),
        timestamp_granularities=timestamp_granularities or None,
    )
    if translate and payload.response_format == "verbose_json":
        # Matches OpenAI's accepted response formats, while still allowing verbose
        # output for models that return segments.
        payload.response_format = "verbose_json"

    data = await upload.read()
    try:
        audio, sample_rate = audio_read(io.BytesIO(data), always_2d=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid audio file: {exc}"
        ) from exc

    return TranscriptionTaskPayload(
        request=payload,
        filename=getattr(upload, "filename", None) or "audio",
        audio=audio,
        sample_rate=sample_rate,
        translate=translate,
    )


def _run_tts_request(request: AudioInferenceRequest) -> None:
    payload: SpeechTaskPayload = request.payload
    speech_request = payload.request
    model, _, _ = get_cached_model(speech_request.model, model_kind="audio_tts")

    ref_audio = speech_request.ref_audio
    if ref_audio and isinstance(ref_audio, str):
        if not os.path.exists(ref_audio):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {ref_audio}",
            )
        from mlx_audio.tts.generate import load_audio

        normalize = hasattr(model, "model_type") and model.model_type == "spark"
        ref_audio = load_audio(
            ref_audio,
            sample_rate=model.sample_rate,
            volume_normalize=normalize,
        )

    generate_kwargs = {
        "voice": speech_request.voice,
        "speed": speech_request.speed,
        "gender": speech_request.gender,
        "pitch": speech_request.pitch,
        "instruct": speech_request.instruct,
        "lang_code": speech_request.lang_code,
        "ref_audio": ref_audio,
        "ref_text": speech_request.ref_text,
        "temperature": speech_request.temperature,
        "top_p": speech_request.top_p,
        "top_k": speech_request.top_k,
        "repetition_penalty": speech_request.repetition_penalty,
        "stream": speech_request.stream,
        "streaming_interval": speech_request.streaming_interval,
        "max_tokens": speech_request.max_tokens,
        "verbose": speech_request.verbose,
    }
    generate_kwargs = _filter_kwargs(model.generate, generate_kwargs)

    audio_chunks = []
    sample_rate = None
    emitted = False
    for result in model.generate(speech_request.input, **generate_kwargs):
        if request.cancel_event.is_set():
            return
        result_audio = getattr(result, "audio", None)
        result_sample_rate = getattr(result, "sample_rate", None)
        if result_audio is None:
            continue
        if result_sample_rate is None:
            result_sample_rate = getattr(model, "sample_rate", None)
        if result_sample_rate is None:
            raise HTTPException(status_code=400, detail="No audio sample rate returned")

        if speech_request.stream:
            encoded = _encode_audio(
                result_audio, result_sample_rate, speech_request.response_format
            )
            if not request.ready_event.is_set():
                request.mark_ready()
            request.emit_data(encoded)
            emitted = True
        else:
            audio_chunks.append(_to_numpy_audio(result_audio))
            if sample_rate is None:
                sample_rate = result_sample_rate

    if speech_request.stream:
        if not emitted:
            raise HTTPException(status_code=400, detail="No audio generated")
        return

    if not audio_chunks or sample_rate is None:
        raise HTTPException(status_code=400, detail="No audio generated")
    audio = audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)
    encoded = _encode_audio(audio, sample_rate, speech_request.response_format)
    if not request.ready_event.is_set():
        request.mark_ready()
    request.emit_data(encoded)


def _run_stt_request(request: AudioInferenceRequest) -> None:
    payload: TranscriptionTaskPayload = request.payload
    transcription_request = payload.request
    model, _, _ = get_cached_model(transcription_request.model, model_kind="audio_stt")

    suffix = os.path.splitext(payload.filename or "")[1] or ".wav"
    tmp_path = None
    emitted = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        audio_write(tmp_path, payload.audio, payload.sample_rate)

        generate_kwargs = _build_stt_generate_kwargs(
            model,
            transcription_request,
            translate=payload.translate,
        )
        result = model.generate(tmp_path, **generate_kwargs)

        for item in _iter_stt_items(result):
            if request.cancel_event.is_set():
                return
            chunk = _stt_item_to_dict(item)
            if not request.ready_event.is_set():
                request.mark_ready()
            request.emit_data(json.dumps(_sanitize_for_json(chunk)) + "\n")
            emitted = True

        if not emitted:
            raise HTTPException(status_code=400, detail="No transcription generated")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_stt_generate_kwargs(
    model,
    request: AudioTranscriptionRequest,
    *,
    translate: bool,
) -> Dict[str, Any]:
    kwargs = request.model_dump(
        exclude={"model", "response_format", "prompt"}, exclude_none=True
    )
    prompt = request.prompt
    accepted = _callable_parameters(model.generate)

    if prompt:
        if _accepts_parameter(accepted, "context"):
            kwargs.setdefault("context", prompt)
        elif _accepts_parameter(accepted, "text"):
            kwargs.setdefault("text", prompt)

    if translate:
        if _accepts_parameter(accepted, "task"):
            kwargs["task"] = "translate"
        elif _accepts_parameter(accepted, "translate"):
            kwargs["translate"] = True
        else:
            raise HTTPException(
                status_code=400,
                detail="Selected model does not expose a translation task parameter.",
            )

    return _filter_kwargs(
        model.generate,
        kwargs,
        extra_allowed={"word_timestamps", "timestamp_granularities"},
    )


def _iter_stt_items(result):
    if isinstance(result, (str, bytes, dict)):
        yield result
        return
    if is_dataclass(result) or hasattr(result, "model_dump") or hasattr(result, "text"):
        yield result
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            yield item
        return
    if hasattr(result, "__iter__"):
        for item in result:
            yield item
        return
    yield result


def _stt_item_to_dict(item) -> Dict[str, Any]:
    if isinstance(item, bytes):
        return {"text": item.decode("utf-8", errors="replace")}
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)

    data = {}
    for attr in ("text", "language", "segments"):
        if hasattr(item, attr):
            data[attr] = getattr(item, attr)
    if hasattr(item, "start_time"):
        data["start"] = getattr(item, "start_time")
    elif hasattr(item, "start"):
        data["start"] = getattr(item, "start")
    if hasattr(item, "end_time"):
        data["end"] = getattr(item, "end_time")
    elif hasattr(item, "end"):
        data["end"] = getattr(item, "end")
    if hasattr(item, "is_final"):
        data["is_final"] = getattr(item, "is_final")
    if not data:
        data["text"] = str(item)
    return data


async def _wait_until_ready(handle: AudioInferenceHandle) -> None:
    await asyncio.to_thread(handle.ready_event.wait)
    if handle.ready_error is not None:
        handle.cancel()
        raise _as_http_exception(handle.ready_error)


async def _collect_audio_handle(handle: AudioInferenceHandle) -> List[Any]:
    chunks = []
    while True:
        chunk = await asyncio.to_thread(handle.result_queue.get)
        if chunk.kind == "data":
            chunks.append(chunk.payload)
        elif chunk.kind == "error":
            handle.cancel()
            raise _as_http_exception(chunk.error)
        elif chunk.kind == "done":
            break
    return chunks


async def _stream_audio_handle(
    handle: AudioInferenceHandle,
    request: Request,
    *,
    endpoint: str,
    model: str,
    request_start: float,
    stream: bool,
    media_kind: str,
):
    completed = False
    try:
        while True:
            if await request.is_disconnected():
                handle.cancel()
                break
            chunk = await asyncio.to_thread(_get_queue_chunk, handle.result_queue, 0.1)
            if chunk is None:
                continue
            if chunk.kind == "data":
                yield chunk.payload
            elif chunk.kind == "error":
                handle.cancel()
                raise _as_http_exception(chunk.error)
            elif chunk.kind == "done":
                completed = True
                break
    except Exception as exc:
        _record_audio_failure(endpoint, model, stream, exc)
        raise
    finally:
        if completed:
            _record_audio_success(
                endpoint=endpoint,
                model=model,
                stream=stream,
                request_start=request_start,
                audio_count=1 if media_kind == "transcription" else 0,
            )


def _get_queue_chunk(result_queue: "queue.Queue[AudioResultChunk]", timeout: float):
    try:
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def _transcription_result_from_chunks(chunks: List[Any]) -> Dict[str, Any]:
    full = None
    accumulated = []
    for chunk in chunks:
        text = (
            chunk.decode("utf-8", errors="replace")
            if isinstance(chunk, bytes)
            else str(chunk)
        )
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if "segments" in obj or "language" in obj:
                full = obj
            if "text" in obj and full is None:
                accumulated.append(obj.get("text") or "")

    if full is None:
        full = {"text": "".join(accumulated).strip()}
    elif "text" not in full:
        full["text"] = "".join(accumulated).strip()
    return full


def _format_transcription_response(result: Dict[str, Any], response_format: str):
    response_format = response_format or "json"
    text = (result.get("text") or "").strip()
    if response_format == "json":
        return JSONResponse({"text": text})
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse(_sanitize_for_json(result))
    if response_format == "srt":
        return PlainTextResponse(_format_segments(result, format="srt"))
    if response_format == "vtt":
        return PlainTextResponse(_format_segments(result, format="vtt"))
    if response_format == "ndjson":
        return PlainTextResponse(json.dumps(_sanitize_for_json(result)) + "\n")
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported transcription response_format: {response_format}",
    )


def _format_segments(result: Dict[str, Any], *, format: Literal["srt", "vtt"]) -> str:
    segments = result.get("segments") or []
    if not segments:
        raise HTTPException(
            status_code=400,
            detail=f"response_format={format} requires timestamped segments.",
        )
    lines = ["WEBVTT", ""] if format == "vtt" else []
    for index, segment in enumerate(segments, start=1):
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "")
        if start is None or end is None:
            raise HTTPException(
                status_code=400,
                detail=f"response_format={format} requires timestamped segments.",
            )
        if format == "srt":
            lines.append(str(index))
            lines.append(f"{_format_time(start, ',')} --> {_format_time(end, ',')}")
        else:
            lines.append(f"{_format_time(start, '.')} --> {_format_time(end, '.')}")
        lines.append(str(text).strip())
        lines.append("")
    return "\n".join(lines)


def _format_time(seconds: float, decimal_separator: str) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{millis:03d}"


def _encode_audio(audio, sample_rate: int, response_format: str) -> bytes:
    buffer = io.BytesIO()
    try:
        audio_write(buffer, _to_numpy_audio(audio), sample_rate, format=response_format)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to encode audio as {response_format}: {exc}",
        ) from exc
    return buffer.getvalue()


def _to_numpy_audio(audio) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio
    return np.array(audio)


def _ensure_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _filter_kwargs(fn, kwargs: Dict[str, Any], extra_allowed=None) -> Dict[str, Any]:
    accepted = _callable_parameters(fn)
    if accepted is None:
        return {k: v for k, v in kwargs.items() if v is not None}
    extra_allowed = set(extra_allowed or ())
    has_var_kwargs = accepted.get("**kwargs", False)
    if has_var_kwargs:
        return {k: v for k, v in kwargs.items() if v is not None}
    params = accepted.get("params", set())
    return {
        k: v
        for k, v in kwargs.items()
        if v is not None and (k in params or k in extra_allowed)
    }


def _callable_parameters(fn):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = set()
    has_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
        else:
            params.add(name)
    return {"params": params, "**kwargs": has_var_kwargs}


def _accepts_parameter(accepted, name: str) -> bool:
    if accepted is None:
        return True
    return accepted.get("**kwargs", False) or name in accepted.get("params", set())


def _sanitize_for_json(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _record_audio_success(
    *,
    endpoint: str,
    model: str,
    stream: bool,
    request_start: float,
    audio_count: int,
) -> None:
    elapsed = time.perf_counter() - request_start
    envelope = _build_metrics_envelope(
        endpoint=endpoint,
        model=model,
        stream=stream,
        backend="audio_queue",
        prompt_tokens=0,
        completion_tokens=0,
        generated_tokens=0,
        request_elapsed_s=elapsed,
        request_started_s=request_start,
        finish_reason="stop",
        audio_count=audio_count,
    )
    runtime.metrics.record_success(envelope)


def _record_audio_failure(
    endpoint: str, model: str, stream: bool, error: BaseException
) -> None:
    runtime.metrics.record_failure(
        endpoint=endpoint,
        model=model,
        stream=stream,
        error=str(getattr(error, "detail", error)),
    )


def _as_http_exception(error: BaseException | None) -> HTTPException:
    if isinstance(error, HTTPException):
        return error
    return HTTPException(status_code=500, detail=str(error))


def _clean_form_value(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    return value if value != "" else None


def _form_list(form, key: str) -> List[str]:
    if not hasattr(form, "getlist"):
        value = form.get(key)
        return [str(value)] if value not in (None, "") else []
    return [str(item) for item in form.getlist(key) if item not in (None, "")]


def _form_bool(value, *, default: bool) -> bool:
    value = _clean_form_value(value)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _form_int(value, *, default: int) -> int:
    value = _clean_form_value(value)
    return default if value is None else int(value)


def _form_float(value, default: Optional[float] = None) -> Optional[float]:
    value = _clean_form_value(value)
    return default if value is None else float(value)
