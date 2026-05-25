"""
Audio feature extractor for Gemma 4.

Ported from HuggingFace Transformers Gemma4AudioFeatureExtractor.
Extracts mel spectrograms from raw audio waveforms using the USM
(Universal Speech Model) preprocessing pipeline.
"""

import math
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np


def _mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.ndarray:
    """Create a mel filter bank matrix (num_frequency_bins, num_mel_filters)."""

    def _hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)
        raise ValueError(f"Unsupported mel_scale: {mel_scale}")

    def _mel_to_hz(mel, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)
        raise ValueError(f"Unsupported mel_scale: {mel_scale}")

    mel_min = _hz_to_mel(min_frequency, mel_scale)
    mel_max = _hz_to_mel(max_frequency, mel_scale)
    mel_points = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    freq_points = np.array([_mel_to_hz(m, mel_scale) for m in mel_points])

    all_freqs = np.arange(num_frequency_bins, dtype=np.float64) * (
        sampling_rate / (2 * (num_frequency_bins - 1))
    )

    filter_bank = np.zeros((num_frequency_bins, num_mel_filters), dtype=np.float64)

    for i in range(num_mel_filters):
        lower = freq_points[i]
        center = freq_points[i + 1]
        upper = freq_points[i + 2]

        # Rising slope
        rising = (all_freqs - lower) / max(center - lower, 1e-10)
        # Falling slope
        falling = (upper - all_freqs) / max(upper - center, 1e-10)

        filter_bank[:, i] = np.maximum(0, np.minimum(rising, falling))

    if norm == "slaney":
        enorm = 2.0 / (freq_points[2:] - freq_points[:-2])
        filter_bank *= enorm[np.newaxis, :]

    return filter_bank.astype(np.float32)


def _unfold(array: np.ndarray, dimension: int, size: int, step: int) -> np.ndarray:
    """A basic NumPy equivalent of PyTorch's unfold for 2D arrays along the last dim."""
    if array.ndim != 2:
        raise ValueError(
            "This unfold implementation currently supports 2D arrays (batch, time)."
        )
    if dimension != -1 and dimension != array.ndim - 1:
        raise ValueError(
            "This unfold implementation only supports unfolding the last dimension."
        )

    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1

    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)

    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])

    return np.lib.stride_tricks.as_strided(
        array, shape=output_shape, strides=output_strides
    )


class Gemma4AudioFeatureExtractor:
    """Audio feature extractor for Gemma 4 using Universal Speech Model preprocessing.

    Extracts log-mel spectrograms from raw audio waveforms with HTK-style
    preemphasis, hanning windowing, and optional per-bin normalization.

    Args:
        feature_size: The feature dimension of the extracted features (num mel bins).
        sampling_rate: The sampling rate in Hz.
        padding_value: Padding value for silence.
        frame_length_ms: The length of a frame in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        min_frequency: Minimum frequency for the mel filterbank.
        max_frequency: Maximum frequency for the mel filterbank.
        preemphasis: The preemphasis coefficient.
        preemphasis_htk_flavor: Whether to use HTK-style preemphasis.
        fft_overdrive: Whether to use FFT overdrive (double FFT length).
        dither: Dithering noise level (0.0 for none).
        input_scale_factor: Scaling factor for input waveform.
        mel_floor: Minimum value for mel spectrograms to avoid log(0).
        per_bin_mean: Mean values for per-bin normalization.
        per_bin_stddev: Standard deviation values for per-bin normalization.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        frame_length_ms: float = 20.0,
        hop_length_ms: float = 10.0,
        min_frequency: float = 0.0,
        max_frequency: float = 8000.0,
        preemphasis: float = 0.0,
        preemphasis_htk_flavor: bool = True,
        fft_overdrive: bool = False,
        dither: float = 0.0,
        input_scale_factor: float = 1.0,
        mel_floor: float = 1e-3,
        per_bin_mean: Optional[Sequence[float]] = None,
        per_bin_stddev: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        self.mel_floor = np.array(mel_floor, dtype=np.float64)

        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length

        # Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2*pi*n / frame_length)
        # Matches HuggingFace Transformers (signal.hann_window with periodic=True)
        self.window = 0.5 - 0.5 * np.cos(
            2.0
            * np.pi
            * np.arange(self.frame_length, dtype=np.float32)
            / self.frame_length
        )

        # Mel filter bank
        try:
            from transformers.audio_utils import mel_filter_bank

            self.mel_filters = mel_filter_bank(
                num_frequency_bins=self.fft_length // 2 + 1,
                num_mel_filters=feature_size,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            )
        except ImportError:
            self.mel_filters = _mel_filter_bank(
                num_frequency_bins=self.fft_length // 2 + 1,
                num_mel_filters=feature_size,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            )

        if per_bin_mean is not None:
            self.per_bin_mean = np.array(per_bin_mean).reshape(1, 1, feature_size)
        else:
            self.per_bin_mean = None

        if per_bin_stddev is not None:
            self.per_bin_stddev = np.array(per_bin_stddev).reshape(1, 1, feature_size)
        else:
            self.per_bin_stddev = None

    def _extract_spectrogram(
        self, waveform: np.ndarray, attention_mask: np.ndarray
    ) -> tuple:
        """Extract log-mel spectrogram from a single waveform."""
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)

        if self.dither > 0.0:
            waveform = waveform + self.dither * np.random.randn(*waveform.shape).astype(
                waveform.dtype
            )

        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        # Semicausal left-padding: prepend frame_length // 2 zeros so that
        # the first frame is centered at t=0, matching HuggingFace Transformers
        pad_left = self.frame_length // 2
        waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode="constant")
        attention_mask = np.pad(
            attention_mask, (pad_left, 0), mode="constant", constant_values=0
        )

        frame_size_for_unfold = self.frame_length + 1

        frames_to_process = _unfold(
            waveform,
            dimension=-1,
            size=frame_size_for_unfold,
            step=self.hop_length,
        )

        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_in_frame = frames_to_process[..., :1] * (1.0 - self.preemphasis)
                rest_in_frame = (
                    frames_to_process[..., 1:-1]
                    - self.preemphasis * frames_to_process[..., :-2]
                )
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = (
                    frames_to_process[..., 1:]
                    - self.preemphasis * frames_to_process[..., :-1]
                )
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)

        magnitude_spec = np.abs(stft)
        mel_spec = np.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = np.log(mel_spec + self.mel_floor)

        if self.per_bin_mean is not None:
            log_mel_spec = log_mel_spec - self.per_bin_mean

        if self.per_bin_stddev is not None:
            log_mel_spec = log_mel_spec / self.per_bin_stddev

        mel_spectrogram = log_mel_spec.squeeze(0)
        num_mel_frames = mel_spectrogram.shape[0]

        frame_end_indices = (
            np.arange(num_mel_frames) * self.hop_length + frame_size_for_unfold - 1
        )
        mask = attention_mask[frame_end_indices].astype(bool)
        return mel_spectrogram, mask

    def _pad_waveforms(self, waveforms, max_length=None, pad_to_multiple_of=None):
        """Pad a list of waveforms to equal length."""
        lengths = [len(w) for w in waveforms]
        target_length = max(lengths)

        if max_length is not None:
            target_length = min(target_length, max_length)

        if pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = (
                (target_length // pad_to_multiple_of) + 1
            ) * pad_to_multiple_of

        padded = []
        masks = []
        for w in waveforms:
            w = np.asarray(w, dtype=np.float32)
            if len(w) > target_length:
                w = w[:target_length]
            mask = np.ones(target_length, dtype=np.int32)
            if len(w) < target_length:
                pad_width = target_length - len(w)
                mask[len(w) :] = 0
                w = np.pad(w, (0, pad_width), constant_values=self.padding_value)
            padded.append(w)
            masks.append(mask)

        return np.array(padded), np.array(masks)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list, list[np.ndarray], list[list[float]]],
        padding: Union[bool, str] = "longest",
        max_length: Optional[int] = 480_000,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = 128,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = True,
        **kwargs,
    ) -> dict:
        """Create a batch of MEL spectrograms from raw speech waveforms.

        Args:
            raw_speech: Single waveform or list of waveforms.
            padding: Padding strategy.
            max_length: Maximum audio length (in samples).
            truncation: Whether to truncate long audio.
            pad_to_multiple_of: Pad to a multiple of this value.
            return_tensors: Not used (kept for API compatibility).
            return_attention_mask: Whether to return attention masks.

        Returns:
            Dict with 'input_features' and 'input_features_mask'.
        """
        # Normalize input to list of 1-D arrays
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1
        is_batched_sequence = isinstance(raw_speech, Sequence) and isinstance(
            raw_speech[0], (np.ndarray, Sequence)
        )
        is_batched = is_batched_numpy or is_batched_sequence

        if not is_batched:
            if not isinstance(raw_speech, np.ndarray):
                raw_speech = np.asarray(raw_speech, dtype=np.float32)
            raw_speech = [raw_speech.flatten()]
        else:
            raw_speech = [
                np.asarray(rs, dtype=np.float32).flatten() for rs in raw_speech
            ]

        # Truncate if needed
        if truncation and max_length is not None:
            raw_speech = [w[:max_length] for w in raw_speech]

        # Pad waveforms
        padded_speech, attention_masks = self._pad_waveforms(
            raw_speech,
            max_length=max_length if padding else None,
            pad_to_multiple_of=pad_to_multiple_of if padding else None,
        )

        # Extract spectrograms
        prepared_speech = []
        prepared_speech_mask = []
        for speech, mask in zip(padded_speech, attention_masks):
            speech_2d = speech.reshape(1, -1)
            spec, spec_mask = self._extract_spectrogram(speech_2d, mask)
            prepared_speech.append(spec.astype(np.float32))
            prepared_speech_mask.append(spec_mask)

        # Zero out padded spectrogram positions, matching HuggingFace Transformers
        prepared_speech = [
            spec * m[..., None]
            for spec, m in zip(prepared_speech, prepared_speech_mask)
        ]

        return {
            "input_features": prepared_speech,
            "input_features_mask": prepared_speech_mask,
        }


__all__ = ["Gemma4AudioFeatureExtractor"]
