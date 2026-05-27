from __future__ import annotations

from collections.abc import Generator
from typing import Any, Tuple

import mlx.core as mx


def generate_step(
    *args: Any, **kwargs: Any
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Compatibility wrapper for the autoregressive generator."""
    from ..generate import generate_step as _generate_step

    yield from _generate_step(*args, **kwargs)
