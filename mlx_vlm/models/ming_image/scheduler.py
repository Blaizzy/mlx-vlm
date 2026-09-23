"""Flow-match Euler scheduler for Ming-Image (dynamic exponential shift).

The checkpoint's ``scheduler_config.json`` stores a static ``shift`` of 6.0, but
the reference pipeline forces ``use_dynamic_shifting`` on and derives a per-image
``mu`` from the latent sequence length instead. This reproduces that behaviour;
the static shift is intentionally unused.
"""

from __future__ import annotations

import mlx.core as mx


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return slope * image_seq_len + (base_shift - slope * base_seq_len)


def resolve_mu(image_seq_len: int) -> float:
    """Ming's runtime rule: high-res prompts (seq >= 4096) clamp mu to 1.35."""
    if image_seq_len >= 4096:
        return calculate_shift(image_seq_len, 256, image_seq_len, 0.5, 1.35)
    return calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)


class FlowMatchEulerDiscreteScheduler:
    def __init__(
        self,
        *,
        image_seq_len: int,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
    ) -> None:
        sigmas = mx.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32
        )
        mu = resolve_mu(image_seq_len)
        sigmas = mx.exp(mu) / (mx.exp(mu) + (1.0 / sigmas - 1.0))
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = mx.concatenate(
            [sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0
        )

    def step(self, *, noise: mx.array, step_index: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[step_index + 1] - self.sigmas[step_index]).astype(noise.dtype)
        return (latents.astype(mx.float32) + dt * noise).astype(noise.dtype)


__all__ = ["FlowMatchEulerDiscreteScheduler", "calculate_shift", "resolve_mu"]
