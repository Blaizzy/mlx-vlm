"""FlowMatchEuler scheduler for Qwen-Image-2.1 (dynamic exponential shift).

Mirrors the structure of ``flux2.scheduler`` but uses Qwen-Image's linear
``calculate_shift`` (base/max shift over image sequence length) instead of
FLUX's empirical mu.
"""

from __future__ import annotations

import mlx.core as mx


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return slope * image_seq_len + (base_shift - slope * base_seq_len)


class FlowMatchEulerDiscreteScheduler:
    def __init__(
        self,
        *,
        image_seq_len: int,
        num_inference_steps: int,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 8192,
        num_train_timesteps: int = 1000,
        shift_terminal: float | None = 0.02,
    ) -> None:
        sigmas = mx.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32
        )
        mu = calculate_shift(
            image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift
        )
        sigmas = mx.exp(mu) / (mx.exp(mu) + (1.0 / sigmas - 1.0))
        # Stretch the shifted schedule before appending the final zero sigma.
        # A one-step schedule has no interval to stretch and starts at pure noise.
        if shift_terminal and num_inference_steps > 1:
            one_minus_sigmas = 1.0 - sigmas
            scale = one_minus_sigmas[-1] / (1.0 - shift_terminal)
            sigmas = 1.0 - one_minus_sigmas / scale
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = mx.concatenate(
            [sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0
        )

    def step(self, *, noise: mx.array, step_index: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[step_index + 1] - self.sigmas[step_index]).astype(noise.dtype)
        return (latents.astype(mx.float32) + dt * noise).astype(noise.dtype)


__all__ = ["FlowMatchEulerDiscreteScheduler", "calculate_shift"]
