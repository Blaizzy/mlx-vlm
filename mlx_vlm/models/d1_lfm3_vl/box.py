# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""Boxes: a presence and four edges, each a distribution over the same grid of fractions of the picture (0 the top
or left side, 1 the bottom or right): the thousandths 0-999 as the tokenizer writes them on a checkpoint that reads
numbers (`edge_grid`), else 128 codes.

- An edge is its distribution's median: an unsure edge's mean drifts to the picture's middle. A checkpoint that
  records `d1_box_edges: interp` reads it inside the median's bin, where the distribution's CDF
  crosses one half (`EDGE_READS`): the bin's value alone is off by a quarter of a bin on average. One without the
  record reads the bin's value.
- A box's confidence: the share of boxes drawn from the edges that overlap it (IoU >= 0.5).
- `boxes`: every instance, blocks of a presence and four edges; block k is present when there are at least k (its
  presence is P(count >= k)), the boxes answered in reading order (left edge, then top). A question reads its `max`
  first blocks, else `INSTANCES`: the count is then at least that when the last is present. Prompt 1 trained 16
  blocks under a prompt saying "16 at most"; prompt 2 states no count and trains 1 to `MAX_INSTANCES` blocks drawn
  per row (`prompt.PROMPTS`).
- Probabilities go through the served calibration's tables (calibration.py)."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable
from functools import lru_cache
from itertools import accumulate

from .grid import Grid

EDGES = ("top", "bottom", "left", "right")
GRID = Grid(0, 1)  # an edge's values as codes, 128 of them
THOUSANDTHS = Grid(
    0, 0.999, 1000
)  # an edge's values as numbers: 0-999 thousandths ("1000" takes two tokens)
DRAWS = 512
INSTANCES = 16  # a `boxes` question's blocks without `max`: 99.97% of COCO val's (picture, class) pairs hold 16 at most
MAX_INSTANCES = 64  # the most a `max` asks, and a prompt-2 training row draws (dense scenes: 17-60 instances)
EDGE_READS = (
    "median",
    "interp",
)  # how an edge is read off its distribution; a checkpoint records it (`d1_box_edges`)


def edge_grid(tokenizer) -> tuple[Grid, int]:
    """An edge's grid and the factor its values are written at: the thousandths where the checkpoint reads numbers
    (`d1_numbers`) and the tokenizer writes each in one token, else the codes."""
    from .prompt import number_ids

    numbers = (
        getattr(tokenizer, "d1_numbers", False)
        and number_ids(tokenizer, THOUSANDTHS, 1000) is not None
    )
    return (THOUSANDTHS, 1000) if numbers else (GRID, 1)


def read_as(distribution) -> Grid:
    """The grid an edge's distribution is over, by its length: the codes' 128 values or the thousandths."""
    return THOUSANDTHS if len(distribution) == THOUSANDTHS.size else GRID


def ordered(a: float, b: float, grid: Grid = GRID) -> tuple[float, float]:
    """Two edges of one axis, the second below or right of the first: crossed ones become one bin about their middle."""
    if a < b:
        return a, b
    half = grid.step / 2
    middle = min(1 - half, max(half, (a + b) / 2))
    return middle - half, middle + half


def rectangle(edges: list[float], grid: Grid = GRID) -> dict[str, float]:
    top, bottom = ordered(*edges[:2], grid)
    left, right = ordered(*edges[2:], grid)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def area(box: dict) -> float:
    return (box["bottom"] - box["top"]) * (box["right"] - box["left"])


def iou(a: dict, b: dict) -> float:
    overlap = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])) * max(
        0.0, min(a["right"], b["right"]) - max(a["left"], b["left"])
    )
    return overlap / (area(a) + area(b) - overlap)


@lru_cache(maxsize=1)
def samples() -> tuple[tuple[float, ...], ...]:
    """`DRAWS` points of a Halton sequence in four dimensions: the same draws every call."""

    def radical_inverse(i: int, base: int) -> float:
        value, scale = 0.0, 1.0
        while i:
            i, digit = divmod(i, base)
            scale /= base
            value += digit * scale
        return value

    return tuple(
        tuple(radical_inverse(i, base) for base in (2, 3, 5, 7))
        for i in range(1, DRAWS + 1)
    )


def confidence(edges: list[list[float]], box: dict) -> float:
    """The share of boxes drawn from the edge distributions (through each one's CDF) with IoU >= 0.5 against `box`."""
    cdfs = [
        list(accumulate(p / total for p in row))
        for row, total in zip(edges, map(math.fsum, edges))
    ]
    grid = read_as(edges[0])
    values = grid.values
    hits = sum(
        iou(
            rectangle(
                [
                    values[min(grid.size - 1, bisect_left(cdf, u))]
                    for cdf, u in zip(cdfs, point)
                ],
                grid,
            ),
            box,
        )
        >= 0.5
        for point in samples()
    )
    return hits / DRAWS


def answer(
    probabilities: list[list[float]],
    through: Callable[[str, float], float] = lambda name, value: value,
    edges: str = "median",
) -> dict:
    """The API's box: `probabilities` are presence ([yes, no]) then the four edges' distributions; `through(name, raw)`
    calibrates `found` ("found:box") and the confidence ("box"); `edges`, how each edge is read (`EDGE_READS`).
    """
    found = through("found:box", probabilities[0][0])
    if found < 0.5:
        return {"type": "box", "box": None, "found": found, "confidence": 0.0}
    box = medians(probabilities[1:], edges)
    return {
        "type": "box",
        "box": box,
        "found": found,
        "confidence": through("box", confidence(probabilities[1:], box)),
    }


def crossing(probabilities, grid: Grid = GRID) -> float:
    """Where the distribution's CDF crosses one half, the mass spread evenly over each bin (value +- step / 2)."""
    total, mass = math.fsum(probabilities), 0.0
    for value, p in zip(grid.values, probabilities):
        share = p / total
        if share > 0 and mass + share >= 0.5:
            return min(
                grid.high,
                max(grid.low, value - grid.step / 2 + (0.5 - mass) / share * grid.step),
            )
        mass += share
    return grid.high


def medians(edges: list[list[float]], read: str = "median") -> dict[str, float]:
    """The box at the four edge distributions' medians: each its bin's value, or where its CDF crosses one half
    inside that bin (`read`)."""
    if read not in EDGE_READS:
        raise ValueError(f"box edges read as one of {EDGE_READS}, not {read!r}")
    grid = read_as(edges[0])
    at = (
        (lambda row: crossing(row, grid))
        if read == "interp"
        else (lambda row: grid.read(row)["median"])
    )
    return rectangle([at(row) for row in edges], grid)


def answer_many(
    probabilities: list[list[float]],
    through: Callable[[str, float], float] = lambda name, value: value,
    edges: str = "median",
) -> dict:
    """The API's `boxes`: `probabilities` are blocks (`INSTANCES`, or a question's `max`) of presence ([yes, no]) and
    four edges. `presence` is P(count >= k) for each k, calibrated ("found:boxes") and kept decreasing; the boxes are
    the blocks at >= 0.5, each with that presence as `found` and its confidence ("boxes"), in reading order; `count`
    is how many, `at_least` when every block read is present (the true count may be higher).
    """
    presence, below = [], 1.0
    for k in range(0, len(probabilities), 5):
        below = min(below, through("found:boxes", probabilities[k][0]))
        presence.append(below)
    out = []
    for k, found in enumerate(presence):
        if found < 0.5:
            break
        block = probabilities[5 * k + 1 : 5 * k + 5]
        box = medians(block, edges)
        out.append(
            {
                "box": box,
                "found": found,
                "confidence": through("boxes", confidence(block, box)),
            }
        )
    out.sort(key=lambda b: (b["box"]["left"], b["box"]["top"]))
    return {
        "type": "boxes",
        "boxes": out,
        "count": len(out),
        "at_least": bool(out) and len(out) == len(presence),
        "presence": presence,
    }
