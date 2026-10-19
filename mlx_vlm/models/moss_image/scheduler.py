from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


class MOSSImageScheduler:
    """Flow matching Euler with MOSS's shifted, nearest training-grid schedule."""

    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        if num_train_timesteps < 1 or not all(
            math.isfinite(s) and s > 0 for s in (shift, training_shift)
        ):
            raise ValueError("Training timesteps and shifts must be positive")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.training_shift = training_shift

    def set_timesteps(self, steps=50):
        if type(steps) is not int or steps < 1:
            raise ValueError("Steps must be a positive integer")
        total = self.num_train_timesteps

        def grid(count, shift):
            values = np.linspace(
                1, shift / (total + shift - 1), count, dtype=np.float32
            )
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(steps, self.shift)
        training = grid(total, self.training_shift)[::-1]
        upper = np.searchsorted(training, sigmas).clip(max=total - 1)
        lower = (upper - 1).clip(min=0)
        nearest = np.where(
            abs(sigmas - training[lower]) <= abs(training[upper] - sigmas), lower, upper
        )
        sigmas = training[nearest]
        sigmas = sigmas[np.r_[True, sigmas[1:] != sigmas[:-1]]].copy()
        self.timesteps = mx.array(sigmas * total)
        self.sigmas = mx.array(np.r_[sigmas, np.float32(0)])

    def step(self, prediction, step_index, latents):
        return latents.astype(mx.float32) + (
            self.sigmas[step_index + 1] - self.sigmas[step_index]
        ) * prediction.astype(mx.float32)
