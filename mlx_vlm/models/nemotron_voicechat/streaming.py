"""Stateful online inference for Nemotron VoiceChat."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
from mlx_audio.codec.models.nemotron_voicechat import CausalConv1dCache
from mlx_audio.stt.models.nemotron_asr import (
    ConformerStreamingState,
    StreamingLogMelSpectrogram,
)
from mlx_audio.stt.models.nemotron_asr import tokenizer as rnnt_tokenizer
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

VoiceChatEventKind = Literal[
    "assistant_text_delta",
    "function_delta",
    "user_transcript_delta",
    "audio",
    "done",
    "cancelled",
]


@dataclass
class VoiceChatEvent:
    """One aligned output from an online VoiceChat session."""

    kind: VoiceChatEventKind
    frame_index: int | None = None
    token_id: int | None = None
    delta: str | None = None
    text: str | None = None
    samples: mx.array | None = None
    sample_rate: int | None = None
    audio_codes: mx.array | None = None


@dataclass(frozen=True)
class VoiceChatFrameTiming:
    """Synchronized stage latency for one native 80 ms input frame."""

    frame_index: int
    perception_ms: float
    rnnt_ms: float
    language_ms: float
    tts_ms: float
    codec_ms: float
    total_ms: float


@dataclass
class VoiceChatProfile:
    """Collected online frame timings and aggregate helpers."""

    frame_duration_ms: float
    frames: list[VoiceChatFrameTiming]

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    def summary(self, *, drop_first: int = 0) -> dict:
        """Return JSON-serializable stage statistics after optional cold frames."""

        frames = self.frames[drop_first:]
        stages = (
            "perception_ms",
            "rnnt_ms",
            "language_ms",
            "tts_ms",
            "codec_ms",
            "total_ms",
        )
        stage_summary = {}
        for stage in stages:
            values = [getattr(frame, stage) for frame in frames]
            stage_summary[stage.removesuffix("_ms")] = {
                "mean_ms": sum(values) / len(values) if values else 0.0,
                "p50_ms": self._percentile(values, 0.5),
                "p95_ms": self._percentile(values, 0.95),
                "max_ms": max(values, default=0.0),
            }
        mean_total = stage_summary["total"]["mean_ms"]
        return {
            "frames": len(frames),
            "dropped_cold_frames": min(drop_first, len(self.frames)),
            "frame_duration_ms": self.frame_duration_ms,
            "processing_frames_per_second": (
                1000.0 / mean_total if mean_total else 0.0
            ),
            "realtime_factor": (
                mean_total / self.frame_duration_ms if self.frame_duration_ms else 0.0
            ),
            "stages": stage_summary,
        }


@dataclass(frozen=True)
class _TimelineTiming:
    language_ms: float
    tts_ms: float
    codec_ms: float


class VoiceChatContextLimitError(RuntimeError):
    pass


class _TokenAccumulator:
    def __init__(self, tokenizer, special_ids: set[int]):
        self.tokenizer = tokenizer
        self.special_ids = special_ids
        self.tokens: list[int] = []
        self.text = ""

    def append(self, token_id: int) -> tuple[str, str] | None:
        if token_id in self.special_ids:
            return None
        self.tokens.append(token_id)
        updated = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        if updated.startswith(self.text):
            delta = updated[len(self.text) :]
        else:
            # Some tokenizers revise a partial byte sequence.  The cumulative
            # value remains authoritative while the one-token decode is still a
            # useful best-effort delta for simple clients.
            delta = self.tokenizer.decode([token_id], skip_special_tokens=False)
        self.text = updated
        return delta, updated


class _RNNTState:
    def __init__(self, parent):
        self.parent = parent
        self.last_token = parent.model.config.rnnt_blank_id
        self.hidden = None
        self.tokens: list[int] = []
        self.text = ""

    def step(self, encoded: mx.array) -> tuple[str, str] | None:
        config = self.parent.model.config
        vocabulary = config.rnnt_vocabulary
        if not vocabulary:
            return None

        decoder = self.parent.model.stt_model.rnnt_decoder
        joint = self.parent.model.stt_model.rnnt_joint
        blank = config.rnnt_blank_id
        new_symbols = 0
        previous_count = len(self.tokens)

        while True:
            token = (
                mx.array([[self.last_token]], dtype=mx.int32)
                if self.last_token != blank
                else None
            )
            prediction, proposed_hidden = decoder(token, self.hidden)
            logits = joint(encoded, prediction.astype(encoded.dtype))
            next_token = int(mx.argmax(logits))
            if next_token == blank:
                break

            self.last_token = next_token
            self.hidden = tuple(
                value.astype(encoded.dtype) for value in proposed_hidden
            )
            if not rnnt_tokenizer.is_special_token(next_token, vocabulary):
                self.tokens.append(next_token)
            new_symbols += 1
            if new_symbols >= config.audio_config.max_symbols:
                break

        if len(self.tokens) == previous_count:
            return None
        updated = rnnt_tokenizer.decode(self.tokens, vocabulary).strip()
        delta = updated[len(self.text) :] if updated.startswith(self.text) else updated
        self.text = updated
        return delta, updated


class VoiceChatStreamingSession:
    """A single stateful, full-duplex VoiceChat timeline.

    Input is buffered into native 80 ms frames. Persistent Nemotron-H, frontend,
    FastConformer, TTS, and codec caches keep each frame's work bounded. The two
    new cache paths can be disabled independently for quality/performance A/Bs.
    """

    def __init__(
        self,
        parent,
        *,
        system_prompt: str | None = None,
        seed: int = 0,
        max_streaming_seconds: float | None = None,
        use_language_cache: bool = True,
        use_perception_cache: bool = True,
        profile: bool = False,
    ):
        if max_streaming_seconds is not None and max_streaming_seconds <= 0:
            raise ValueError("max_streaming_seconds must be positive")

        self.parent = parent
        self.model = parent.model
        self.tokenizer = parent.tokenizer
        self.config = self.model.config
        self.input_sample_rate = self.config.input_sample_rate
        self.output_sample_rate = self.config.output_sample_rate
        self.frame_samples = round(self.input_sample_rate * self.config.frame_duration)
        self.max_frames = (
            None
            if max_streaming_seconds is None
            else int(max_streaming_seconds / self.config.frame_duration)
        )
        self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._audio_window = mx.zeros((0,), dtype=mx.float32)
        left, right = self.config.audio_config.encoder.att_context_size[0]
        self._perception_window_frames = max(2, left + right + 1)
        self.use_language_cache = use_language_cache
        self.use_perception_cache = use_perception_cache
        self._language_cache = (
            self.model.stt_model.make_cache() if use_language_cache else None
        )
        self._mel_stream = None
        self._conformer_stream = None
        if use_perception_cache:
            self._mel_stream = StreamingLogMelSpectrogram(
                self.config.audio_config.preprocessor,
                lookahead_samples=self.frame_samples,
            )
            self._conformer_stream = ConformerStreamingState(
                self.model.stt_model.perception.encoder,
                chunk_frames=1,
                att_context_size=[left, right],
            )
        self._input_history: list[mx.array] = []
        self._text_tokens: list[int] = []
        self._function_tokens: list[int] = []
        self._frame_index = 0
        self._timeline_index = 0
        self._closed = False
        self.profile = VoiceChatProfile(
            frame_duration_ms=self.config.frame_duration * 1000.0,
            frames=[],
        )
        self._profiling = profile

        special_ids = {
            self.config.pad_token_id,
            self.config.silence_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        self._text = _TokenAccumulator(self.tokenizer, special_ids)
        self._function = _TokenAccumulator(self.tokenizer, special_ids)
        self._rnnt = _RNNTState(parent)
        self._codec_cache = CausalConv1dCache()

        mx.random.seed(seed)
        prompt = self.parent._tts_prompt()
        _, self._tts_cache = self.model.tts_model.tts_model.warmup(
            *prompt[:-1],
            audio_prompt_latent=prompt[-1],
            guidance=True,
        )
        self._previous_code = prompt[0][:, -1:]
        self._prefill_prompt(system_prompt)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def _language_step(self, inputs: mx.array):
        if self._language_cache is not None:
            return self.model.stt_model(inputs, cache=self._language_cache)
        self._input_history.append(inputs)
        return self.model.stt_model(mx.concatenate(self._input_history, axis=1))

    def _prefill_prompt(self, system_prompt: str | None) -> None:
        prompt_text = (
            self.config.default_system_prompt
            if system_prompt is None
            else system_prompt
        )
        if not prompt_text.strip():
            return

        prompt_ids = [
            self.config.bos_token_id,
            *self.tokenizer.encode(prompt_text, add_special_tokens=False),
            self.config.eos_token_id,
        ]
        prompt_embeds = self.model.stt_model.embed_tokens(
            mx.array([prompt_ids], dtype=mx.int32)
        )
        for index in range(len(prompt_ids)):
            self._run_timeline_step(
                prompt_embeds[:, index : index + 1],
                generate_channels=False,
                decode_audio=False,
            )

    def _perception_step(self, frame: mx.array) -> tuple[mx.array, mx.array]:
        if self._mel_stream is not None and self._conformer_stream is not None:
            mel = self._mel_stream.push(frame)
            chunks = self._conformer_stream.push(mel, emit_partial=True)
            if len(chunks) != 1 or chunks[0].shape[1] != 1:
                shapes = [tuple(chunk.shape) for chunk in chunks]
                raise RuntimeError(
                    "cached perception did not emit exactly one encoder frame: "
                    f"mel={tuple(mel.shape)}, encoded={shapes}"
                )
            encoded = chunks[0]
            projected = self.model.stt_model.perception.proj(encoded)
            self._conformer_stream.materialize(projected, encoded)
            return projected, encoded

        self._audio_window = mx.concatenate([self._audio_window, frame])
        max_samples = self._perception_window_frames * self.frame_samples
        if self._audio_window.shape[0] > max_samples:
            self._audio_window = self._audio_window[-max_samples:]

        mel = log_mel_spectrogram(
            self._audio_window,
            self.config.audio_config.preprocessor,
        )
        mel_lengths = mx.array([mel.shape[1]], dtype=mx.int32)
        projected, _, encoded = self.model.stt_model.perception(mel, mel_lengths)
        if projected.shape[1] < 2:
            raise RuntimeError(
                "perception encoder returned fewer than two frames for an 80 ms input"
            )
        # This is the reference non-cache streaming rule.  The final embedding
        # contains the preprocessor's right-edge silence padding.
        projected = projected[:, -2:-1]
        encoded = encoded[:, -2:-1]
        mx.eval(projected, encoded)
        return projected, encoded

    def _run_timeline_step(
        self,
        audio_embedding: mx.array,
        *,
        generate_channels: bool,
        decode_audio: bool,
    ) -> tuple[list[VoiceChatEvent], _TimelineTiming]:
        pad_id = self.config.pad_token_id
        previous_text_id = (
            pad_id if self._timeline_index == 0 else self._text_tokens[-1]
        )
        previous_function_id = (
            pad_id if self._timeline_index == 0 else self._function_tokens[-1]
        )
        previous_text = mx.array([[previous_text_id]], dtype=mx.int32)
        previous_function = mx.array([[previous_function_id]], dtype=mx.int32)
        fused = (
            self.model.stt_model.embed_tokens(previous_text)
            + audio_embedding
            + self.config.function_channel_weight
            * self.model.stt_model.embed_tokens(previous_function)
        )
        stage_start = time.perf_counter()
        output = self._language_step(fused)
        text_id = (
            int(mx.argmax(output.text_logits[:, -1])) if generate_channels else pad_id
        )
        function_id = (
            int(mx.argmax(output.function_logits[:, -1]))
            if generate_channels
            else pad_id
        )
        if not generate_channels:
            mx.eval(output.text_logits, output.function_logits)
        language_ms = (time.perf_counter() - stage_start) * 1000.0
        self._text_tokens.append(text_id)
        self._function_tokens.append(function_id)

        stage_start = time.perf_counter()
        if self._timeline_index == 0:
            code = self.model.tts_model.codec_silence_tokens[None, None, :]
        else:
            current = mx.array([[text_id]], dtype=mx.int32)
            if text_id == self.config.eos_token_id:
                self._previous_code = mx.broadcast_to(
                    self.model.tts_model.codec_silence_tokens[None, None, :],
                    self._previous_code.shape,
                )
            tts_output = self.model.tts_model.tts_model.step(
                self._previous_code,
                current,
                mx.ones(current.shape, dtype=mx.bool_),
                self._tts_cache,
                guidance=True,
            )
            self._previous_code = tts_output.codes
            code = self._previous_code

        self._timeline_index += 1
        mx.eval(code)
        tts_ms = (time.perf_counter() - stage_start) * 1000.0
        if not generate_channels:
            return [], _TimelineTiming(language_ms, tts_ms, 0.0)

        events: list[VoiceChatEvent] = []
        text_update = self._text.append(text_id)
        if text_update is not None:
            delta, text = text_update
            events.append(
                VoiceChatEvent(
                    kind="assistant_text_delta",
                    frame_index=self._frame_index,
                    token_id=text_id,
                    delta=delta,
                    text=text,
                )
            )
        function_update = self._function.append(function_id)
        if function_update is not None:
            delta, text = function_update
            events.append(
                VoiceChatEvent(
                    kind="function_delta",
                    frame_index=self._frame_index,
                    token_id=function_id,
                    delta=delta,
                    text=text,
                )
            )

        codec_ms = 0.0
        if decode_audio:
            stage_start = time.perf_counter()
            clean_code = self.parent._replace_control_codes(code)
            samples = self.model.tts_model.audio_codec.decode_step(
                clean_code.transpose(0, 2, 1),
                self._codec_cache,
            )[0, 0]
            mx.eval(samples)
            codec_ms = (time.perf_counter() - stage_start) * 1000.0
            expected_samples = self.model.tts_model.audio_codec.waveform_to_token_ratio
            if samples.shape[0] != expected_samples:
                raise RuntimeError(
                    "streaming codec returned an unexpected number of samples: "
                    f"{samples.shape[0]}"
                )
            events.append(
                VoiceChatEvent(
                    kind="audio",
                    frame_index=self._frame_index,
                    samples=samples,
                    sample_rate=self.output_sample_rate,
                    audio_codes=clean_code[0, 0],
                )
            )
        return events, _TimelineTiming(language_ms, tts_ms, codec_ms)

    def _step_audio_frame(self, frame: mx.array) -> list[VoiceChatEvent]:
        if self.max_frames is not None and self._frame_index >= self.max_frames:
            raise VoiceChatContextLimitError(
                f"stream exceeded the configured {self.max_frames}-frame context"
            )
        frame_start = time.perf_counter()
        stage_start = frame_start
        projected, encoded = self._perception_step(frame)
        perception_ms = (time.perf_counter() - stage_start) * 1000.0
        events: list[VoiceChatEvent] = []
        stage_start = time.perf_counter()
        transcript_update = self._rnnt.step(encoded)
        rnnt_ms = (time.perf_counter() - stage_start) * 1000.0
        if transcript_update is not None:
            delta, text = transcript_update
            events.append(
                VoiceChatEvent(
                    kind="user_transcript_delta",
                    frame_index=self._frame_index,
                    delta=delta,
                    text=text,
                )
            )
        timeline_events, timeline_timing = self._run_timeline_step(
            projected,
            generate_channels=True,
            decode_audio=True,
        )
        events.extend(timeline_events)
        total_ms = (time.perf_counter() - frame_start) * 1000.0
        if self._profiling:
            self.profile.frames.append(
                VoiceChatFrameTiming(
                    frame_index=self._frame_index,
                    perception_ms=perception_ms,
                    rnnt_ms=rnnt_ms,
                    language_ms=timeline_timing.language_ms,
                    tts_ms=timeline_timing.tts_ms,
                    codec_ms=timeline_timing.codec_ms,
                    total_ms=total_ms,
                )
            )
        self._frame_index += 1
        return events

    def push_audio(
        self,
        samples,
        *,
        sample_rate: int | None = None,
    ) -> list[VoiceChatEvent]:
        """Consume arbitrary mono float PCM chunks and emit completed frames."""

        if self._closed:
            raise RuntimeError("streaming session is closed")
        sample_rate = self.input_sample_rate if sample_rate is None else sample_rate
        if sample_rate != self.input_sample_rate:
            raise ValueError(
                f"expected {self.input_sample_rate} Hz PCM, received {sample_rate} Hz"
            )
        chunk = mx.array(samples, dtype=mx.float32)
        if chunk.ndim == 2 and 1 in chunk.shape:
            chunk = chunk.reshape(-1)
        if chunk.ndim != 1:
            raise ValueError("audio must be mono PCM")
        if chunk.shape[0] == 0:
            return []

        self._pending_audio = mx.concatenate([self._pending_audio, chunk])
        events: list[VoiceChatEvent] = []
        while self._pending_audio.shape[0] >= self.frame_samples:
            frame = self._pending_audio[: self.frame_samples]
            self._pending_audio = self._pending_audio[self.frame_samples :]
            events.extend(self._step_audio_frame(frame))
        return events

    def flush(self, *, pad_partial: bool = True) -> list[VoiceChatEvent]:
        """Finish the input stream, optionally zero-padding its partial frame."""

        if self._closed:
            return []
        events: list[VoiceChatEvent] = []
        if self._pending_audio.shape[0] and pad_partial:
            frame = mx.pad(
                self._pending_audio,
                (0, self.frame_samples - self._pending_audio.shape[0]),
            )
            self._pending_audio = mx.zeros((0,), dtype=mx.float32)
            events.extend(self._step_audio_frame(frame))
        else:
            self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._closed = True
        self._codec_cache.clear()
        events.append(VoiceChatEvent(kind="done", frame_index=self._frame_index))
        return events

    def cancel(self) -> list[VoiceChatEvent]:
        if self._closed:
            return []
        self._closed = True
        self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._codec_cache.clear()
        return [VoiceChatEvent(kind="cancelled", frame_index=self._frame_index)]
