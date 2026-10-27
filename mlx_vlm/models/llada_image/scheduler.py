from __future__ import annotations

import mlx.core as mx
import numpy as np


class LLaDAImageScheduler:
    def __init__(self, config: dict):
        self.use_uniform_sigmas = bool(config.get("use_uniform_sigmas", False))
        if config.get("use_dynamic_shifting", False) or config.get(
            "invert_sigmas", False
        ):
            raise ValueError(
                "LLaDA-Image requires a static, non-inverted flow schedule"
            )
        self.shift = float(config.get("shift", 3.0 if self.use_uniform_sigmas else 1.0))
        self.stochastic_sampling = bool(
            config.get("stochastic_sampling", self.use_uniform_sigmas)
        )

    @property
    def default_steps(self) -> int:
        return 4 if self.use_uniform_sigmas else 50

    @property
    def default_guidance(self) -> float:
        return 1.0 if self.use_uniform_sigmas else 5.0

    def sigmas(self, steps: int) -> mx.array:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if self.use_uniform_sigmas:
            sigma = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)
        else:
            # The base checkpoint constructs its nonlinear schedule in float64.
            schedule = np.linspace(0.001, 1.0, steps + 1)[:-1]
            sigma = 1.0 - (1.0 - (1.0 - schedule**1.17) ** 0.8) ** 1.1
            sigma = mx.array(np.append(sigma, 0.0).astype(np.float32))
        return self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)

    def step(self, prediction, sample, sigma, sigma_next, *, noise=None):
        sample = sample.astype(mx.float32)
        prediction = prediction.astype(mx.float32)
        if self.stochastic_sampling:
            if noise is None:
                noise = mx.random.normal(sample.shape, dtype=mx.float32)
            clean = sample - sigma * prediction
            return (1.0 - sigma_next) * clean + sigma_next * noise
        return sample + (sigma_next - sigma) * prediction
