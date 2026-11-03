"""Distributed-training helpers for the trainer backend.

Keep MLX process-group initialization, deterministic rank sharding, collective
operations, and rank-zero checkpoint ownership here. Modality trainer modules
should not call ``mx.distributed`` directly.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class DistributedContext:
    """Immutable view of the MLX process group for a training run."""

    group: Any
    rank: int
    world_size: int


def initialize() -> DistributedContext:
    """Initialize MLX distributed execution once and expose its stable metadata."""
    group = mx.distributed.init()
    return DistributedContext(group=group, rank=group.rank(), world_size=group.size())


def all_sum(value: mx.array) -> mx.array:
    """Reduce a scalar or array across the initialized training group."""
    return mx.distributed.all_sum(value, stream=mx.cpu)
