# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""Ordered values read as a distribution over equally spaced bins."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence


@dataclass(frozen=True)
class Grid:
    low: float
    high: float
    size: int = 128

    def __post_init__(self):
        if self.size < 2 or not self.low < self.high:
            raise ValueError("a grid needs two values or more, low to high")

    @property
    def step(self) -> float:
        return (self.high - self.low) / (self.size - 1)

    @cached_property
    def values(self) -> tuple[float, ...]:
        return tuple(self.low + i * self.step for i in range(self.size - 1)) + (
            self.high,
        )

    def read(
        self,
        probabilities: Sequence[float],
        level: Callable[[float], float] | None = None,
    ) -> dict[str, float]:
        """The mean, median and central 80 percent interval, on this grid's values; `level`, the level a quantile is
        read at (a calibration's, increasing)."""
        total = math.fsum(probabilities)
        values = self.values
        out = {
            "mean": math.fsum(v * (p / total) for v, p in zip(values, probabilities))
        }
        at = [
            (key, level(q) if level else q)
            for key, q in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
        ]
        mass = 0.0
        for value, p in zip(values, probabilities):
            mass += p / total
            for key, quantile in at:
                if key not in out and mass >= quantile:
                    out[key] = value
        # a level near 1 the summed mass falls short of by rounding: the last value
        for key, _ in at:
            out.setdefault(key, values[-1])
        return out
