"""Recording-wide diarization and speaker vectors on the shared audio worker."""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import time
from dataclasses import dataclass
from typing import BinaryIO

import mlx.core as mx
import numpy as np
from fastapi import HTTPException, Request

DIARIZATION_MODEL = "nativ-community/Nemotron-3-Diarization"
EMBEDDING_MODEL = "nativ-community/SpeechBrain-ECAPA-VoxCeleb-20M"
EMBEDDING_DIMENSION = 192
SAMPLE_RATE = 16000
READ_SAMPLES = 30 * SAMPLE_RATE
MAX_WINDOWS = 3
MAX_SECONDS = 30


@dataclass
class SpeakerTaskPayload:
    audio: BinaryIO
    sample_count: int
    revision: str | None
    windows: list[tuple[int, int]] | None = None

    def close(self):
        self.audio.close()

    def read_window(self, start: int, end: int) -> np.ndarray:
        self.audio.seek(start * 4)
        return np.frombuffer(self.audio.read((end - start) * 4), dtype="<f4")

    def chunks(self, check_cancelled):
        self.audio.seek(0)
        while data := self.audio.read(READ_SAMPLES * 4):
            check_cancelled()
            chunk = np.frombuffer(data, dtype="<f4")
            if not np.isfinite(chunk).all():
                raise ValueError("The recording contains invalid audio samples")
            yield chunk


def register_routes(app):
    for path, endpoint in (
        ("diarizations", audio_diarizations_endpoint),
        ("speaker_embeddings", audio_speaker_embeddings_endpoint),
    ):
        app.post(f"/audio/{path}", response_model=None)(endpoint)
        app.post(f"/v1/audio/{path}", response_model=None, include_in_schema=False)(
            endpoint
        )


async def audio_diarizations_endpoint(request: Request):
    return await _response(request, "diarization")


async def audio_speaker_embeddings_endpoint(request: Request):
    return await _response(request, "speaker_embedding")


async def _response(request: Request, kind: str):
    from .audio import _audio_analysis_events

    endpoint = (
        "/v1/audio/diarizations"
        if kind == "diarization"
        else "/v1/audio/speaker_embeddings"
    )
    started = time.perf_counter()
    model, payload = await parse_speaker_request(request, kind)
    async with _audio_analysis_events(
        request,
        kind=kind,
        model=model,
        payload=payload,
        endpoint=endpoint,
        request_start=started,
    ) as events:
        result = None
        async for result in events:
            pass
        if result is None:
            raise HTTPException(500, "Audio analysis returned no result")
        return result


async def parse_speaker_request(
    request: Request, kind: str, *, model_key="model", revision_key="revision"
):
    audio = tempfile.NamedTemporaryFile()
    try:
        async with request.form() as form:
            default_model = (
                DIARIZATION_MODEL if kind == "diarization" else EMBEDDING_MODEL
            )
            model = form.get(model_key, default_model)
            revision = form.get(revision_key) or None
            if (
                not isinstance(model, str)
                or not model.strip()
                or (revision is not None and not isinstance(revision, str))
            ):
                raise ValueError("Invalid model or revision")
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise ValueError("Missing file")
            # PCM uploads allow long recordings to stay on disk; no JSON array
            # or base64 copy of a multi-hour meeting is retained in memory.
            input_format = form.get("input_format", "audio")
            maximum = (
                (8 * 3600 if kind == "diarization" else MAX_SECONDS) * SAMPLE_RATE * 4
            )
            size = 0
            while data := await upload.read(1024 * 1024):
                size += len(data)
                if size > (
                    maximum
                    if input_format == "pcm_f32le"
                    else max(maximum, 32 * 1024 * 1024)
                ):
                    raise ValueError("Audio exceeds the endpoint duration/size limit")
                audio.write(data)
            audio.seek(0)
            if input_format == "pcm_f32le":
                if form.get("sample_rate", "16000") != "16000":
                    raise ValueError("PCM must be mono float32 at 16000 Hz")
            elif input_format == "audio":
                from mlx_audio.audio_io import read
                from scipy.signal import resample_poly

                audio.flush()
                try:
                    samples, rate = await asyncio.to_thread(
                        read, audio.name, always_2d=False
                    )
                except Exception as exc:
                    raise ValueError(f"Invalid audio file: {exc}") from exc
                samples = np.asarray(samples, dtype=np.float32)
                if samples.ndim == 2:
                    samples = samples.mean(axis=1)
                if samples.ndim != 1 or rate <= 0:
                    raise ValueError("Invalid audio format")
                if rate != SAMPLE_RATE:
                    divisor = math.gcd(rate, SAMPLE_RATE)
                    samples = await asyncio.to_thread(
                        resample_poly, samples, SAMPLE_RATE // divisor, rate // divisor
                    )
                size = samples.size * 4
                if size > maximum:
                    raise ValueError("Audio exceeds the endpoint duration limit")
                audio.seek(0)
                audio.truncate()
                audio.write(np.asarray(samples, dtype="<f4").tobytes())
            else:
                raise ValueError("input_format must be audio or pcm_f32le")
            if size < 512 * 4 or size % 4:
                raise ValueError("Invalid audio buffer")
            windows = None
            if kind == "speaker_embedding":
                windows = validate_windows(
                    json.loads(form.get("windows", "[]")), size // 4
                )
            audio.seek(0)
            return model, SpeakerTaskPayload(audio, size // 4, revision, windows)
    except BaseException as exc:
        audio.close()
        if isinstance(exc, (ValueError, TypeError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise


def load_speaker_model(model_id: str, kind: str, revision: str | None):
    if kind == "audio_diarization":
        from mlx_audio.vad import load

        return load(model_id, revision=revision, strict=True)
    from mlx_audio.codec import EcapaSpeakerEmbeddingExtractor

    return EcapaSpeakerEmbeddingExtractor.from_pretrained(model_id, revision=revision)


def run_request(request, get_cached_model):
    payload = request.payload
    try:
        # Validate all samples before loading, so malformed late blocks never
        # generate partial output or trigger a model download.
        for _ in payload.chunks(request.check_cancelled):
            pass
        if request.kind == "speaker_embedding":
            for start, end in payload.windows:
                window = payload.read_window(start, end)
                if float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) < 1e-4:
                    raise ValueError("The selected speaker audio is too quiet")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.check_cancelled()
    model, _, _ = get_cached_model(
        request.model_name,
        model_kind=f"audio_{request.kind}",
        revision=payload.revision,
    )
    request.check_cancelled()
    request.mark_ready()
    if request.kind == "diarization":
        result = {"segments": diarize_segments(request, model)}
    else:
        total = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        sample_count = 0
        for start, end in payload.windows:
            request.check_cancelled()
            window = payload.read_window(start, end).copy()
            vector = model.encode(window, sample_rate=SAMPLE_RATE)
            mx.eval(vector)
            total += normalized(vector) * len(window)
            sample_count += len(window)
        result = {
            "embedding": normalized(total).tolist(),
            "duration": sample_count / SAMPLE_RATE,
        }
    request.check_cancelled()
    request.emit_data(
        {"model": request.model_name, "revision": payload.revision, **result}
    )


def validate_windows(windows: list, sample_count: int) -> list[tuple[int, int]]:
    if not isinstance(windows, list) or not 1 <= len(windows) <= MAX_WINDOWS:
        raise ValueError("Expected one to three single-speaker audio windows.")
    validated = []
    previous_end = 0
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("Invalid audio window.")
        start, end = window.get("start"), window.get("end")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (start, end)
        ):
            raise ValueError("Audio window times must be finite.")
        if start < 0 or end > sample_count / SAMPLE_RATE:
            raise ValueError("Audio window is outside the speaker audio buffer.")
        first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        length = last - first
        if (
            first < previous_end
            or last > sample_count
            or not 3 * SAMPLE_RATE <= length <= 10 * SAMPLE_RATE
        ):
            raise ValueError(
                "Audio windows must be separate and between three and ten seconds."
            )
        validated.append((first, last))
        previous_end = last
    if sum(end - start for start, end in validated) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError("Too much audio for a speaker embedding.")
    return validated


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
        raise ValueError("The speaker encoder returned an invalid embedding.")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("The speaker encoder returned an empty embedding.")
    return vector / norm


def diarize_segments(request, model):
    """One model stream owns the complete recording and its speaker cache."""
    payload = request.payload
    duration = payload.sample_count / SAMPLE_RATE
    segments, last_by_speaker = [], {}
    # One generator per recording, never one per upload block or ASR chunk.
    # MLX-Audio owns the recording-wide speaker cache and absolute clock.
    stream = model.generate_stream(
        payload.chunks(request.check_cancelled), sample_rate=SAMPLE_RATE
    )
    try:
        for result in stream:
            request.check_cancelled()
            for turn in result.segments:
                start, end, speaker = (
                    float(turn.start),
                    float(turn.end),
                    int(turn.speaker),
                )
                if not math.isfinite(start) or not math.isfinite(end) or speaker < 0:
                    raise ValueError("Invalid diarization output")
                start, end = max(0.0, start), min(duration, end)
                if end <= start:
                    continue
                previous = last_by_speaker.get(speaker)
                if previous is not None and start <= previous["end"] + 1e-6:
                    previous["end"] = max(previous["end"], end)
                else:
                    segment = {"start": start, "end": end, "speaker": speaker}
                    segments.append(segment)
                    last_by_speaker[speaker] = segment
    finally:
        stream.close()
    return sorted(segments, key=lambda item: (item["start"], item["speaker"]))
