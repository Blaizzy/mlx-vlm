"""Drafter registry for speculative decoding.

Each drafter kind maps to a loader that returns a ready-to-use drafter
module. Today only ``dflash`` (block-diffusion drafter for Qwen3.5) is
registered; EAGLE/Medusa/etc. can be added by appending to ``_LOADERS``.
"""

from typing import Any, Callable, Dict

import mlx.core as mx

from .qwen3_5_dflash import DFlashDraftModel, load_dflash_drafter

_LOADERS: Dict[str, Callable[..., Any]] = {
    "dflash": load_dflash_drafter,
}


def load_drafter(
    path_or_repo: str,
    kind: str = "dflash",
    dtype: mx.Dtype = mx.bfloat16,
    **kwargs,
):
    """Load a speculative-decoding drafter.

    Args:
        path_or_repo: Local path or HuggingFace repo id for the drafter.
        kind: Drafter family. Currently only ``"dflash"`` is supported.
        dtype: Weight dtype to cast to at load time.
        **kwargs: Forwarded to the underlying loader.
    """
    if kind not in _LOADERS:
        raise ValueError(
            f"Unknown drafter kind {kind!r}. Known: {sorted(_LOADERS)}"
        )
    return _LOADERS[kind](path_or_repo, dtype=dtype, **kwargs)


__all__ = ["DFlashDraftModel", "load_dflash_drafter", "load_drafter"]
