"""Drafter registry for speculative decoding.

Today only ``dflash`` (block-diffusion drafter for Qwen3.5) is registered;
EAGLE/Medusa/etc. can be added later. Loading goes through the shared
:func:`mlx_vlm.utils.load_model`, which picks up drafters via the extra
import path in ``utils._EXTRA_MODEL_PACKAGES`` and the ``dflash_config``
key detection in ``get_model_and_args``.
"""

from .qwen3_dflash import DFlashDraftModel


def load_drafter(path_or_repo: str, kind: str = "dflash", **kwargs):
    """Load a speculative-decoding drafter via :func:`mlx_vlm.utils.load_model`.

    Args:
        path_or_repo: Local path or HuggingFace repo id for the drafter.
        kind: Drafter family. Currently only ``"dflash"`` is supported.
            Dispatch is actually driven by ``get_model_and_args`` inside
            ``load_model``, so this argument is retained for validation.
        **kwargs: Forwarded to ``load_model`` (e.g. ``lazy=True``).
    """
    if kind != "dflash":
        raise ValueError(f"Unknown drafter kind {kind!r}. Known: ['dflash']")
    from ...utils import get_model_path, load_model

    return load_model(get_model_path(path_or_repo), **kwargs)


__all__ = ["DFlashDraftModel", "load_drafter"]
