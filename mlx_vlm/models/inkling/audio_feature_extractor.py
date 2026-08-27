from collections.abc import Sequence
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from mlx_audio.dsp import hanning, mel_filters, stft


def _exact_sample_count(value: float, name: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


class InklingAudioFeatureExtractor:
    """Convert waveforms to the log-mel frames consumed by Inkling's dMel tokenizer."""

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_token_duration_s: float = 0.05,
        window_size_multiplier: float = 2.0,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_size: Optional[int] = None,
        max_frames_per_chunk: int = 1024,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        computed_hop = _exact_sample_count(
            audio_token_duration_s * sampling_rate,
            "audio_token_duration_s * sampling_rate",
        )
        computed_window = _exact_sample_count(
            audio_token_duration_s * window_size_multiplier * sampling_rate,
            "audio_token_duration_s * window_size_multiplier * sampling_rate",
        )

        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.audio_token_duration_s = audio_token_duration_s
        self.window_size_multiplier = window_size_multiplier
        self.hop_length = computed_hop if hop_length is None else hop_length
        self.window_size = computed_window if window_size is None else window_size
        self.n_fft = self.window_size if n_fft is None else n_fft
        self.max_frames_per_chunk = max_frames_per_chunk
        self.return_attention_mask = return_attention_mask

        if min(self.hop_length, self.window_size, self.n_fft) <= 0:
            raise ValueError("hop_length, window_size, and n_fft must be positive")
        if self.window_size > self.n_fft:
            raise ValueError("window_size cannot be larger than n_fft")
        if self.max_frames_per_chunk <= 0:
            raise ValueError("max_frames_per_chunk must be positive")

        window = hanning(self.window_size, periodic=True).astype(mx.float32)
        if self.window_size < self.n_fft:
            left = (self.n_fft - self.window_size) // 2
            right = self.n_fft - self.window_size - left
            window = mx.pad(window, [(left, right)])
        self.window = window
        self.mel_filters = mel_filters(
            sampling_rate,
            self.n_fft,
            feature_size,
            norm="slaney",
            mel_scale="slaney",
            precise=True,
        )

    @staticmethod
    def _as_clips(raw_speech) -> list:
        if isinstance(raw_speech, (np.ndarray, mx.array)):
            if raw_speech.ndim > 2:
                raise ValueError(
                    "A single audio array must be 1-D or 2-D; pass a list of "
                    "arrays for a batch."
                )
            return [raw_speech]

        if not isinstance(raw_speech, (list, tuple)):
            raise TypeError(f"Unsupported audio input type: {type(raw_speech)}")
        if not raw_speech:
            raise ValueError("Received an empty audio input.")
        if np.isscalar(raw_speech[0]):
            return [raw_speech]
        return list(raw_speech)

    @staticmethod
    def _to_mono(clip) -> mx.array:
        if isinstance(clip, mx.array):
            waveform = clip.astype(mx.float32)
        else:
            waveform = mx.array(np.asarray(clip, dtype=np.float32))

        if waveform.ndim == 2:
            waveform = mx.mean(waveform, axis=-1)
        elif waveform.ndim != 1:
            raise ValueError(
                f"Each audio clip must be 1-D or 2-D, got shape {waveform.shape}."
            )
        if waveform.size == 0:
            raise ValueError("Audio clips must contain at least one sample.")
        return waveform

    def _extract_log_mel(self, waveform: mx.array, padded_length: int) -> mx.array:
        right_pad = (-padded_length) % self.hop_length
        left_pad = max(self.n_fft - self.hop_length, 0)
        total_length = left_pad + padded_length + right_pad
        num_frames = 1 + (total_length - self.n_fft) // self.hop_length

        chunks = []
        for frame_start in range(0, num_frames, self.max_frames_per_chunk):
            frame_count = min(self.max_frames_per_chunk, num_frames - frame_start)
            segment_start = frame_start * self.hop_length - left_pad
            segment_length = (frame_count - 1) * self.hop_length + self.n_fft
            segment_end = segment_start + segment_length

            data_start = min(max(segment_start, 0), waveform.shape[0])
            data_end = min(max(segment_end, data_start), waveform.shape[0])
            segment = waveform[data_start:data_end]
            pad_left = max(-segment_start, 0)
            pad_right = segment_length - pad_left - segment.shape[0]

            if pad_left or pad_right:
                segment = mx.pad(segment, [(pad_left, pad_right)])
            spectrum = stft(
                segment,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
            )
            magnitudes = mx.maximum(mx.abs(spectrum), 1e-10)
            # vmap uses accurate GEMV reductions; Metal GEMM can move values
            # across nearby dMel bin boundaries.
            mel = mx.vmap(lambda mel_filter: magnitudes @ mel_filter)(
                self.mel_filters
            ).T
            log_mel = mx.log10(mx.maximum(mel, 1e-10))

            # Bound the lazy graph and release FFT temporaries before the next chunk.
            mx.eval(log_mel)
            chunks.append(log_mel)

        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)

    def __call__(
        self,
        raw_speech: Union[
            np.ndarray,
            mx.array,
            Sequence[float],
            Sequence[np.ndarray],
        ],
        sampling_rate: Optional[int] = None,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Inkling expects {self.sampling_rate} Hz audio, got {sampling_rate} Hz."
            )

        waveforms = [self._to_mono(clip) for clip in self._as_clips(raw_speech)]
        if truncation and max_length is not None:
            waveforms = [waveform[:max_length] for waveform in waveforms]
            if any(waveform.size == 0 for waveform in waveforms):
                raise ValueError("max_length must retain at least one audio sample.")

        lengths = [waveform.shape[0] for waveform in waveforms]
        if padding in (True, "longest"):
            padded_length = max(lengths)
        elif padding == "max_length":
            if max_length is None:
                raise ValueError("padding='max_length' requires max_length")
            padded_length = max_length
        elif padding in (False, None, "do_not_pad"):
            if len(set(lengths)) != 1:
                raise ValueError(
                    "Batched audio with different lengths requires padding"
                )
            padded_length = lengths[0]
        else:
            raise ValueError(f"Unsupported padding strategy: {padding}")

        if any(length > padded_length for length in lengths):
            raise ValueError("Audio longer than max_length requires truncation=True")

        if pad_to_multiple_of:
            padded_length = (
                (padded_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        features = []
        masks = []
        for waveform, length in zip(waveforms, lengths):
            mel = self._extract_log_mel(waveform, padded_length)
            valid_frames = (length + self.hop_length - 1) // self.hop_length
            mask = mx.arange(mel.shape[0]) < valid_frames
            mel = mx.where(mask[:, None], mel, self.padding_value)
            features.append(mel.astype(mx.float32))
            masks.append(mask)

        result = {"input_features": mx.stack(features)}
        include_mask = (
            self.return_attention_mask
            if return_attention_mask is None
            else return_attention_mask
        )
        if include_mask:
            result["input_features_mask"] = mx.stack(masks)
        return result


__all__ = ["InklingAudioFeatureExtractor"]
