from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


class MiniMaxH3Scheduler:
    """MiniMax-H3's shifted rectified-flow Euler scheduler."""

    order = 1

    def __init__(self, shift: float = 12.0) -> None:
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)
        self.num_inference_steps: int | None = None
        self.sigmas: mx.array | None = None
        self.timesteps: mx.array | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        sigmas: Sequence[float] | mx.array | None = None,
    ) -> None:
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "set_timesteps requires explicit sigmas or "
                    f"num_inference_steps >= 2, got {num_inference_steps}"
                )
            base = mx.linspace(
                1.0,
                0.0,
                int(num_inference_steps),
                dtype=mx.float32,
            )
            schedule = self._shift * base / (1.0 + (self._shift - 1.0) * base)
            # MLX does not support boolean array indexing. The schedule is a
            # small control tensor, so collapse float32 collisions on the host
            # exactly as torch.unique_consecutive does and return to MLX.
            schedule_values = schedule.tolist()
            schedule = mx.array(
                [
                    value
                    for index, value in enumerate(schedule_values)
                    if index == 0 or value != schedule_values[index - 1]
                ],
                dtype=mx.float32,
            )
        else:
            schedule = mx.array(sigmas, dtype=mx.float32).reshape(-1)
            if schedule.size < 2:
                raise ValueError(
                    "sigmas must hold at least two strictly decreasing values "
                    "ending at 0.0"
                )
            values = schedule.tolist()
            if values[-1] != 0.0 or any(
                right >= left for left, right in zip(values, values[1:])
            ):
                raise ValueError(
                    "sigmas must hold at least two strictly decreasing values "
                    "ending at 0.0"
                )

        self.sigmas = schedule
        self.timesteps = 1.0 - schedule[:-1]
        self.num_inference_steps = int(self.timesteps.size)
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | mx.array) -> int:
        if self.timesteps is None:
            raise RuntimeError("set_timesteps must be called before index_for_timestep")
        value = float(timestep.item()) if isinstance(timestep, mx.array) else timestep
        indices = [
            index
            for index, candidate in enumerate(self.timesteps.tolist())
            if candidate == value
        ]
        if not indices:
            raise ValueError(
                "timestep is not in scheduler.timesteps; pass a value from the schedule"
            )
        return indices[0]

    def scale_noise(
        self,
        sample: mx.array,
        timestep: float | mx.array,
        noise: mx.array,
    ) -> mx.array:
        time = mx.array(timestep, dtype=sample.dtype)
        while time.ndim < sample.ndim:
            time = mx.expand_dims(time, axis=-1)
        return time * sample + (1.0 - time) * noise

    def step(
        self,
        model_output: mx.array,
        timestep: float | mx.array,
        sample: mx.array,
    ) -> mx.array:
        if self.sigmas is None or self.timesteps is None:
            raise RuntimeError("set_timesteps must be called before step")
        if isinstance(timestep, int) or (
            isinstance(timestep, mx.array)
            and not mx.issubdtype(timestep.dtype, mx.floating)
        ):
            raise ValueError(
                "integer step indices are unsupported; pass a value from "
                "scheduler.timesteps"
            )
        if self._step_index is None:
            self._step_index = (
                self.index_for_timestep(timestep)
                if self._begin_index is None
                else self._begin_index
            )

        time = mx.array(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1.0 - time
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = mx.expand_dims(sigma_from_timestep, axis=-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = (
            mx.float32 if sample.dtype in (mx.float16, mx.bfloat16) else sample.dtype
        )
        sigma = self.sigmas[self._step_index].astype(compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].astype(compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = (
            ratio * sample.astype(compute_dtype)
            + (1.0 - ratio) * denoised.astype(compute_dtype)
        ).astype(sample.dtype)
        self._step_index += 1
        return prev_sample


__all__ = ["MiniMaxH3Scheduler"]
