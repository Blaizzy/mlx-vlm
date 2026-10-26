"""Diarized file transcription with finalized OpenAI SSE segment events.

Inference windows are independent of speaker turns. A single diarization stream
owns all speaker IDs; overlapping ASR windows are aligned before publication.
"""

import json
import math
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .audio_speakers import (
    SAMPLE_RATE,
    SpeakerTaskPayload,
    diarize_segments,
    parse_speaker_request,
)

if TYPE_CHECKING:
    from .audio import AudioTranscriptionRequest


@dataclass(kw_only=True)
class DiarizedTranscriptionPayload(SpeakerTaskPayload):
    transcription: "AudioTranscriptionRequest"
    diarization_model: str
    include_diarization: bool


async def response(request):
    from .audio import AudioTranscriptionRequest, _audio_analysis_events, _form_bool

    form = await request.form()
    model = form.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "Missing model.")
    # Do not silently accept options whose semantics we cannot implement.
    for key in (
        "known_speaker_names",
        "known_speaker_references",
        "timestamp_granularities",
    ):
        if key in form or f"{key}[]" in form:
            raise HTTPException(
                400, f"{key} is not supported for local diarized transcription"
            )
    strategy = form.get("chunking_strategy", "auto")
    if strategy != "auto":
        raise HTTPException(400, 'Only chunking_strategy="auto" is currently supported')
    stream = _form_bool(form.get("stream"), default=False)
    include = form.getlist("include[]") + form.getlist("include")
    if any(value != "diarization" for value in include):
        raise HTTPException(400, "Only include[]=diarization is supported")
    stt = AudioTranscriptionRequest(
        model=model,
        language=form.get("language"),
        prompt=form.get("prompt"),
        word_timestamps=True,
        stream=False,
    )
    diarization_model, payload = await parse_speaker_request(
        request,
        "diarization",
        model_key="diarization_model",
        revision_key="diarization_revision",
    )
    payload = DiarizedTranscriptionPayload(
        audio=payload.audio,
        sample_count=payload.sample_count,
        revision=payload.revision,
        transcription=stt,
        diarization_model=diarization_model,
        include_diarization="diarization" in include,
    )
    events = _audio_analysis_events(
        request,
        kind="diarized_stt",
        model=model,
        payload=payload,
        endpoint="/v1/audio/transcriptions",
        stream=stream,
    )

    if stream:

        async def sse():
            try:
                async with events as source:
                    async for event in source:
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                error = {
                    "type": "error",
                    "error": {"message": str(getattr(exc, "detail", exc))},
                }
                yield f"data: {json.dumps(error)}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    async with events as source:
        segments, result = [], None
        async for event in source:
            if event["type"] == "transcript.text.segment":
                segments.append(event)
            elif event["type"] == "transcript.text.done":
                result = {key: value for key, value in event.items() if key != "type"}
        if result is None:
            raise HTTPException(500, "Transcription ended without a final result")
        return {**result, "segments": segments}


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    audio_start: float
    audio_end: float


def transcription_windows(segments, duration):
    """18–26 second cores, silence cuts, 2 seconds of context, <=30s ASR."""
    rate = SAMPLE_RATE
    total, context = round(duration * rate), 2 * rate
    maximum, minimum, preferred, room = 26 * rate, 18 * rate, 24 * rate, rate // 4
    speech = []
    for segment in sorted(segments, key=lambda s: s["start"]):
        start, end = round(segment["start"] * rate), round(segment["end"] * rate)
        if speech and start <= speech[-1][1]:
            speech[-1] = (speech[-1][0], max(speech[-1][1], end))
        else:
            speech.append((start, end))
    groups = []
    for span in speech:
        if groups and span[0] - groups[-1][-1][1] <= 2 * rate:
            groups[-1].append(span)
        else:
            groups.append([span])
    windows = []
    for group in groups:
        group_start, group_end = max(0, group[0][0] - room), min(
            total, group[-1][1] + room
        )
        silence = [
            (a[1], b[0]) for a, b in zip(group, group[1:]) if b[0] - a[1] >= room
        ]
        start = group_start
        while start < group_end:
            remaining = group_end - start
            end = min(group_end, start + maximum)
            if remaining > maximum:
                lower, upper, target = (
                    start + minimum,
                    start + maximum,
                    start + preferred,
                )
                if remaining <= 2 * maximum and remaining - maximum < minimum:
                    lower = start + min(minimum, remaining // 2)
                    end = target = start + remaining // 2
                if remaining <= 2 * maximum:
                    lower = max(lower, group_end - maximum)
                tail = min(minimum, remaining // 2) if remaining <= 2 * maximum else 0
                latest = min(upper, group_end - tail)
                candidates = [
                    min(min(latest, b), max(max(lower, a), (a + b) // 2))
                    for a, b in silence
                    if max(lower, a) <= min(latest, b)
                ]
                if candidates:
                    end = min(candidates, key=lambda cut: (abs(cut - target), cut))
            windows.append(
                Window(
                    start / rate,
                    end / rate,
                    max(group_start, start - context) / rate,
                    min(group_end, end + context) / rate,
                )
            )
            start = end
    return windows


@dataclass
class Word:
    text: str
    start: float
    end: float
    joins_previous: bool = False

    @property
    def midpoint(self):
        return (self.start + self.end) / 2


def aligned_words(result, duration):
    """Normalize Parakeet token pieces or explicit word timestamps, never guess."""
    from .audio import _stt_item_to_dict

    result = _stt_item_to_dict(result)
    text = result.get("text", "").strip()
    if result.get("words") is None and result.get("segments"):
        timed_segments = result["segments"]
        if all(s.get("words") is not None for s in timed_segments):
            result["words"] = [w for s in timed_segments for w in s["words"]]
    tokens = result.get("tokens")
    if not tokens and result.get("sentences"):
        tokens = [
            token
            for sentence in result["sentences"]
            for token in sentence.get("tokens", [])
        ]
    words = []
    if tokens:
        for token in tokens:
            if not isinstance(token, dict):
                raise ValueError(
                    "The transcription model must return aligned word or token timestamps"
                )
            piece, start = token["text"], float(token["start"])
            end = float(token.get("end", start + token.get("duration", 0)))
            if words and piece and not piece[0].isspace():
                words[-1].text += piece
                words[-1].end = max(words[-1].end, end)
            else:
                words.append(Word(piece, start, end))
    elif result.get("words") is not None:
        words = [
            Word(w.get("word", w.get("text", "")), float(w["start"]), float(w["end"]))
            for w in result["words"]
        ]
    elif text:
        raise ValueError(
            "diarized_json requires a model with word timestamps (for example Parakeet)"
        )
    words = [Word(w.text.strip(), w.start, w.end) for w in words if w.text.strip()]
    cursor, previous = 0, -1.0
    for i, word in enumerate(words):
        if not (
            math.isfinite(word.start)
            and math.isfinite(word.end)
            and 0 <= word.start <= word.end <= duration + 0.4
            and word.start < duration + 0.08
            and word.start >= previous
        ):
            raise ValueError("Invalid transcription word timestamps")
        at = text.find(word.text, cursor)
        if at < 0 or text[cursor:at].strip():
            raise ValueError("Word timestamps do not cover the transcription text")
        word.joins_previous = i > 0 and at == cursor
        cursor, previous = at + len(word.text), word.start
    if text[cursor:].strip():
        raise ValueError("Word timestamps do not cover the transcription text")
    return words


def stitch(accumulated, incoming, window, previous):
    """Align the unpublished overlap, preserving match and timestamp tie-breaking."""
    if (
        previous is None
        or previous.end != window.start
        or previous.audio_end <= window.audio_start
    ):
        accumulated.extend(
            w for w in incoming if window.start <= w.midpoint < window.audio_end
        )
        return
    lo, hi = max(previous.audio_start, window.audio_start), min(
        previous.audio_end, window.audio_end
    )
    left = [i for i, w in enumerate(accumulated) if lo <= w.midpoint < hi]
    right = [i for i, w in enumerate(incoming) if lo <= w.midpoint < hi]
    # Lexicographic score: maximize monotonic matches, minimize timestamp drift.
    scores = [[(0, 0.0) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]
    steps = {}
    normalize = lambda w: "".join(c for c in w.text.lower() if c.isalnum())
    left_text = [normalize(accumulated[i]) for i in left]
    right_text = [normalize(incoming[i]) for i in right]
    for i, li in enumerate(left, 1):
        for j, ri in enumerate(right, 1):
            best, step = scores[i - 1][j], 1
            if scores[i][j - 1] > best:
                best, step = scores[i][j - 1], 2
            a, b = accumulated[li], incoming[ri]
            difference = abs(a.midpoint - b.midpoint)
            if (
                left_text[i - 1]
                and left_text[i - 1] == right_text[j - 1]
                and difference <= 0.75
            ):
                before = scores[i - 1][j - 1]
                match = (before[0] + 1, before[1] - difference)
                if match > best:
                    best, step = match, 3
            scores[i][j], steps[i, j] = best, step
    matches, i, j = [], len(left), len(right)
    while i and j:
        step = steps[i, j]
        if step == 3:
            matches.append((left[i - 1], right[j - 1]))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    if matches:
        a, b = min(
            matches,
            key=lambda pair: (
                abs(
                    (accumulated[pair[0]].midpoint + incoming[pair[1]].midpoint) / 2
                    - window.start
                ),
                pair[0],
            ),
        )
        accumulated[a + 1 :] = incoming[b + 1 :]
    else:
        accumulated[:] = [w for w in accumulated if w.midpoint < window.start]
        accumulated.extend(
            w for w in incoming if window.start <= w.midpoint < window.audio_end
        )


def speakers_for_word(word, segments):
    active = sorted(
        {s["speaker"] for s in segments if s["start"] <= word.midpoint < s["end"]}
    )
    if active:
        return active
    overlaps = [
        (s["speaker"], max(0, min(word.end, s["end"]) - max(word.start, s["start"])))
        for s in segments
    ]
    longest = max((v for _, v in overlaps), default=0)
    if longest > 0:
        return sorted({s for s, v in overlaps if abs(v - longest) < 1 / SAMPLE_RATE})
    before = [s for s in segments if s["end"] <= word.start]
    after = [s for s in segments if s["start"] >= word.end]
    if before and after:
        end, start = max(s["end"] for s in before), min(s["start"] for s in after)
        prev, nxt = {s["speaker"] for s in before if s["end"] == end}, {
            s["speaker"] for s in after if s["start"] == start
        }
        if start - end <= 2 and prev == nxt:
            return sorted(prev)
    distances = [
        (s["speaker"], max(s["start"] - word.end, word.start - s["end"], 0))
        for s in segments
    ]
    closest = min((d for _, d in distances), default=math.inf)
    nearest = {s for s, d in distances if abs(d - closest) < 1 / SAMPLE_RATE}
    return sorted(nearest) if closest <= 0.25 and len(nearest) == 1 else []


def speaker_label(index):
    label = ""
    while index >= 0:
        label = chr(65 + index % 26) + label
        index = index // 26 - 1
    return label


def run_request(request, get_cached_model):
    from scipy.io.wavfile import write

    from .audio import _build_stt_generate_kwargs

    payload = request.payload
    for _ in payload.chunks(request.check_cancelled):
        pass  # Validate the complete source before publishing anything.
    diarizer, _, _ = get_cached_model(
        payload.diarization_model,
        model_kind="audio_diarization",
        revision=payload.revision,
    )
    request.check_cancelled()
    segments = diarize_segments(request, diarizer)
    duration = payload.sample_count / SAMPLE_RATE
    labels = {
        speaker: speaker_label(i)
        for i, speaker in enumerate(sorted({s["speaker"] for s in segments}))
    }
    metadata = {
        "model": payload.diarization_model,
        "revision": payload.revision,
        "segments": segments,
        "speaker_ids": {label: speaker for speaker, label in labels.items()},
    }
    request.mark_ready()
    if payload.include_diarization:
        request.emit_data({"type": "diarization.completed", "diarization": metadata})
    windows = transcription_windows(segments, duration)
    words, previous, segment_count, text_parts = [], None, 0, []
    model = None
    if windows:
        model, _, _ = get_cached_model(request.model_name, model_kind="audio_stt")
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/chunk.wav"
        for index, window in enumerate(windows):
            request.check_cancelled()
            start, end = round(window.audio_start * SAMPLE_RATE), round(
                window.audio_end * SAMPLE_RATE
            )
            samples = payload.read_window(start, end).copy()
            if len(samples) < 512:
                samples = np.pad(samples, (0, 512 - len(samples)))
            # Float WAV preserves the original PCM; MLX-Audio's generic writer
            # quantizes float samples to int16, which is unnecessary for ASR.
            write(path, SAMPLE_RATE, samples)
            kwargs = _build_stt_generate_kwargs(
                model, payload.transcription, translate=False
            )
            result = model.generate(path, **kwargs)
            request.check_cancelled()
            incoming = aligned_words(result, window.audio_end - window.audio_start)
            for word in incoming:
                word.start = min(
                    window.audio_end - 1 / SAMPLE_RATE, window.audio_start + word.start
                )
                word.end = min(window.audio_end, window.audio_start + word.end)
            stitch(words, incoming, window, previous)
            previous = window
            # The next window may replace only words inside shared context.
            # Publish the stable prefix and retain that tail until it is aligned.
            cutoff = (
                windows[index + 1].audio_start if index + 1 < len(windows) else math.inf
            )
            stable = 0
            while stable < len(words) and words[stable].midpoint < cutoff:
                stable += 1
            turns = []
            for word in words[:stable]:
                ids = speakers_for_word(word, segments)
                text_parts.append(
                    ("" if not text_parts or word.joins_previous else " ") + word.text
                )
                if (
                    turns
                    and turns[-1]["speakers"] == ids
                    and word.start - turns[-1]["end"] <= 1
                    and word.end - turns[-1]["start"] <= 30
                ):
                    turns[-1]["text"] += (
                        "" if word.joins_previous else " "
                    ) + word.text
                    turns[-1]["end"] = max(turns[-1]["end"], word.end)
                else:
                    turns.append(
                        {
                            "start": word.start,
                            "end": max(word.end, word.start + 1 / SAMPLE_RATE),
                            "text": word.text,
                            "speakers": ids,
                        }
                    )
            for turn in turns:
                ids = turn.pop("speakers")
                event = {
                    "type": "transcript.text.segment",
                    "id": f"seg_{segment_count:06d}",
                    **turn,
                    "speaker": labels[ids[0]] if ids else "unknown",
                }
                if payload.include_diarization:
                    event["speakers"] = [labels[s] for s in ids]
                request.emit_data(event)
                segment_count += 1
            # Published words cannot participate in the next overlap. Release
            # them so alignment work and retained timings stay window-sized.
            del words[:stable]
    done = {
        "type": "transcript.text.done",
        "task": "transcribe",
        "duration": duration,
        "text": "".join(text_parts),
    }
    if payload.include_diarization:
        done["diarization"] = metadata
    request.check_cancelled()
    request.emit_data(done)
