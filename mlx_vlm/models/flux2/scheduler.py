from __future__ import annotations

import mlx.core as mx


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, *, image_seq_len: int, num_inference_steps: int) -> None:
        self.timesteps, self.sigmas = self.get_timesteps_and_sigmas(
            image_seq_len=image_seq_len,
            num_inference_steps=num_inference_steps,
        )

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def _time_shift_exponential_array(
        mu: float, sigma_power: float, t: mx.array
    ) -> mx.array:
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    @classmethod
    def get_timesteps_and_sigmas(
        cls,
        *,
        image_seq_len: int,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
    ) -> tuple[mx.array, mx.array]:
        sigmas = mx.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32
        )
        mu = cls._compute_empirical_mu(
            image_seq_len=image_seq_len, num_steps=num_inference_steps
        )
        sigmas = cls._time_shift_exponential_array(mu, 1.0, sigmas)
        timesteps = sigmas * num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        return timesteps, sigmas

    def step(self, *, noise: mx.array, step_index: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[step_index + 1] - self.sigmas[step_index]).astype(
            latents.dtype
        )
        return latents + dt * noise.astype(latents.dtype)
