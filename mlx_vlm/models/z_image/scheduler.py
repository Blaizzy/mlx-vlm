"""Flow-match Euler scheduler for Z-Image."""

from __future__ import annotations

import mlx.core as mx


class FlowMatchEulerScheduler:
    """Simple flow-matching Euler scheduler.

    Timestep convention: the model receives (1000 - t) / 1000, where t comes
    from a linearly-spaced schedule in [0, 1000].
    """

    def __init__(self, num_inference_steps: int) -> None:
        if num_inference_steps < 1:
            raise ValueError(f"steps must be >= 1, got {num_inference_steps}")
        self.num_steps = num_inference_steps
        # Linearly spaced from ~1000 to ~0
        self.timesteps = mx.linspace(
            1000.0, 1000.0 / num_inference_steps, num_inference_steps
        )
        # sigma = t / 1000
        self.sigmas = self.timesteps / 1000.0
        self.sigmas = mx.concatenate([self.sigmas, mx.zeros((1,))])

    def get_model_input_timestep(self, step_index: int) -> mx.array:
        """Returns the normalized timestep the model expects."""
        t = self.timesteps[step_index]
        return mx.array([(1000.0 - t.item()) / 1000.0])

    def step(self, noise_pred: mx.array, step_index: int, latents: mx.array) -> mx.array:
        """Euler step: latents + dt * noise_pred (negated in pipeline)."""
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        dt = sigma_next - sigma
        return latents + dt.astype(latents.dtype) * noise_pred


__all__ = ["FlowMatchEulerScheduler"]
