from pathlib import Path

import mlx.core as mx


class StepAudio2Vocoder:
    """Lazy, MLX-native waveform decoding using the mlx-audio speech codec."""

    sample_rate = 24000

    def __init__(self, *, n_timesteps: int = 10):
        try:
            from mlx_audio.codec.models.stepaudio2 import StepAudio2Token2Wav
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-o waveform decoding requires mlx-audio with the "
                "StepAudio2 codec. Install mlx-audio>=0.4.8."
            ) from exc

        self.n_timesteps = n_timesteps
        self._codec = StepAudio2Token2Wav.from_pretrained()

    def decode(self, audio_tokens: mx.array, *, prompt_wav_path: str) -> mx.array:
        if prompt_wav_path is None or not Path(prompt_wav_path).is_file():
            raise ValueError("The StepAudio2 codec requires a reference audio file.")
        if (
            audio_tokens.ndim != 3
            or audio_tokens.shape[0] != 1
            or audio_tokens.shape[2] != 1
        ):
            raise ValueError("StepAudio2 expects audio tokens with shape (1, time, 1).")
        if audio_tokens.shape[1] == 0:
            raise ValueError("No speech tokens were generated.")
        wav = self._codec(
            audio_tokens[0, :, 0],
            prompt_wav_path,
            n_timesteps=self.n_timesteps,
            # mlx-audio's prompt cache is not keyed by voice. Reusing the codec
            # must not reuse another request's reference speaker.
            use_cache=False,
        )
        return wav.reshape(-1).astype(mx.float32)
