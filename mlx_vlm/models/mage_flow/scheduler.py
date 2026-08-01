from __future__ import annotations

import mlx.core as mx


class FlowMatchEulerDiscreteScheduler:
    """Static-shift flow-matching Euler scheduler used by Mage-Flow."""

    def __init__(self, *, num_inference_steps: int, shift: float = 6.0) -> None:
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
        base = mx.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=mx.float32,
        )
        self.sigmas = shift * base / (1.0 + (shift - 1.0) * base)
        self.timesteps = self.sigmas * 1000.0
        self.sigmas = mx.concatenate(
            [self.sigmas, mx.zeros((1,), dtype=mx.float32)], axis=0
        )

    def step(self, *, velocity: mx.array, step_index: int, latents: mx.array):
        dt = (self.sigmas[step_index + 1] - self.sigmas[step_index]).astype(
            latents.dtype
        )
        return latents + dt * velocity.astype(latents.dtype)


__all__ = ["FlowMatchEulerDiscreteScheduler"]
