from __future__ import annotations

import mlx.core as mx


class ErnieImageFlowMatchScheduler:
    def __init__(
        self,
        *,
        num_inference_steps: int,
        shift: float = 4.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
        sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1, dtype=mx.float32)[:-1]
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        self.timesteps = sigmas * float(num_train_timesteps)
        self.sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=mx.float32)], axis=0)

    def step(
        self, *, model_output: mx.array, step_index: int, sample: mx.array
    ) -> mx.array:
        sample_dtype = sample.dtype
        sample = sample.astype(mx.float32)
        delta = self.sigmas[step_index + 1] - self.sigmas[step_index]
        sample = sample + delta * model_output.astype(mx.float32)
        return sample.astype(sample_dtype)


__all__ = ["ErnieImageFlowMatchScheduler"]
