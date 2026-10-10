"""Stateless configuration and batching helpers shared by trainer modules."""

from .common.utils import (
    calculate_iters,
    pad_arrays,
    resolve_args,
    resolve_pad_token_id,
)

__all__ = ["calculate_iters", "pad_arrays", "resolve_args", "resolve_pad_token_id"]
