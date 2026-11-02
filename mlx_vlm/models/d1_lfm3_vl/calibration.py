# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""Temperature scaling of option scores, and tables for the typed answers' probabilities.

- `calibration.json` holds a global temperature and optional per-type ones; scaling keeps an argmax but can move a
  score's median or expected level.
- A typed probability goes through its table, a monotone map fitted on held-out rows: `found:text`,
  `found:number`, `found:date`, `found:box`, `box` (a box's confidence), `found:boxes` (P(count >= k)), `boxes`
  (each listed box's confidence), `multiple` (each option's P(yes)), `estimate` (a quantile level, read backwards by
  `Calibration.level`).
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from .prompt import Noul, Question, Score, cardinality


def bucket(kind: str, k: int) -> str:
    if kind == "noul":
        return "noul"
    if kind == "score":
        return f"score:{k}"
    if k <= 3:
        return "choice:small"
    if k <= 8:
        return "choice:mid"
    if k <= 26:
        return "choice:large"
    return "choice:inventory"


def question_bucket(q: Question) -> str:
    kind = (
        "noul" if isinstance(q, Noul) else "score" if isinstance(q, Score) else "choice"
    )
    return bucket(kind, cardinality(q))


def through(value: float, knots: list[list[float]] | None) -> float:
    """`value` through increasing (raw, calibrated) knots, linear between them, flat past the ends."""
    if not knots:
        return value
    index = bisect_left([raw for raw, _ in knots], value)
    if index == 0:
        return knots[0][1]
    if index == len(knots):
        return knots[-1][1]
    (a, pa), (b, pb) = knots[index - 1], knots[index]
    return pa + (value - a) * (pb - pa) / (b - a)


@dataclass
class Calibration:
    temperatures: dict[str, float] = field(default_factory=dict)
    default: float = 1.0
    counts: dict[str, int] = field(default_factory=dict)
    tables: dict[str, list[list[float]]] = field(default_factory=dict)

    def temperature(self, q: Question) -> float:
        return self.temperatures.get(question_bucket(q), self.default)

    def apply(self, q: Question, scores: list[float]) -> list[float]:
        t = self.temperature(q)
        return scores if t == 1.0 else [s / t for s in scores]

    def through(self, name: str, value: float) -> float:
        return through(value, self.tables.get(name))

    def level(self, name: str, share: float) -> float:
        """The level whose observed share is `share`: the table read backwards (both columns increase)."""
        return through(share, [[y, x] for x, y in self.tables.get(name) or ()])

    @classmethod
    def load(cls, path: Path | str) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            d["temperatures"],
            d.get("default", 1.0),
            d.get("counts", {}),
            d.get("tables", {}),
        )
