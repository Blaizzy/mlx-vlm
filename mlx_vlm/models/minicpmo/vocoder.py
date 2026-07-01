from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


def _tokens_to_list(audio_tokens) -> list[int]:
    if isinstance(audio_tokens, mx.array):
        arr = np.array(audio_tokens)
    else:
        arr = np.asarray(audio_tokens)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr[..., 0]
    return [int(x) for x in arr.reshape(-1).tolist()]


class StepAudio2Vocoder:
    """Wrapper around the native mlx-audio StepAudio2 codec."""

    def __init__(
        self,
        *,
        n_timesteps: int = 10,
    ):
        try:
            from mlx_audio.codec.models.stepaudio2 import StepAudio2Token2Wav
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-o native waveform decoding requires a version of "
                "`mlx-audio` that exposes `mlx_audio.codec.models.stepaudio2`."
            ) from exc

        self.n_timesteps = n_timesteps
        self._codec = StepAudio2Token2Wav.from_pretrained()

    def decode(
        self,
        audio_tokens,
        *,
        prompt_wav_path: str,
        output_audio_path: Optional[str] = None,
    ) -> bytes:
        if prompt_wav_path is None:
            raise ValueError("The native stepaudio2 codec requires a prompt WAV path.")

        wav = self._codec(
            mx.array(_tokens_to_list(audio_tokens), dtype=mx.int32),
            prompt_wav_path,
            n_timesteps=self.n_timesteps,
        )
        wav_bytes = self._codec.to_wav_bytes(wav)
        if output_audio_path is not None:
            Path(output_audio_path).write_bytes(wav_bytes)
        return wav_bytes
