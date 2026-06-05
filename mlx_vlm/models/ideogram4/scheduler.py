from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True, slots=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: float) -> float:
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        if t <= 0.0:
            return t_max
        if t >= 1.0:
            return t_min
        z = NormalDist().inv_cdf(t)
        shifted = self.mean + self.std * z
        value = 1.0 - (1.0 / (1.0 + math.exp(-shifted)))
        return min(max(value, t_min), t_max)


@dataclass(frozen=True, slots=True)
class SamplerPreset:
    num_steps: int
    guidance_schedule: tuple[float, ...]
    mu: float
    std: float = 1.0

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError("guidance_schedule length must match num_steps")


PRESETS: dict[str, SamplerPreset] = {
    "V4_QUALITY_48": SamplerPreset(
        num_steps=48,
        guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
        mu=0.0,
        std=1.5,
    ),
    "V4_DEFAULT_20": SamplerPreset(
        num_steps=20,
        guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
        mu=0.0,
        std=1.75,
    ),
    "V4_TURBO_12": SamplerPreset(
        num_steps=12,
        guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
        mu=0.5,
        std=1.75,
    ),
}


def get_preset(name: str | None) -> SamplerPreset:
    key = name or "V4_DEFAULT_20"
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Ideogram 4 sampler preset {name!r}; "
            f"expected one of {sorted(PRESETS)}"
        ) from exc


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 1.0,
    std: float = 1.0,
) -> LogitNormalSchedule:
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> tuple[float, ...]:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return tuple(i / num_steps for i in range(num_steps + 1))
