"""MLX port of the DFlash block-diffusion drafter.

Exports the ``Model`` / ``ModelConfig`` attributes that
:func:`mlx_vlm.utils.load_model` looks up, so the drafter loads through
the shared ``load_model`` entry point. The drafter has no vision /
language / audio towers, so the ``VisionModel`` / ``LanguageModel`` /
``AudioModel`` attributes aren't defined — ``load_model`` already
guards those lookups with ``hasattr``.
"""

from .config import DFlashConfig as ModelConfig
from .dflash import DFlashDraftModel
from .dflash import DFlashDraftModel as Model
from .dflash import DFlashKVCache

__all__ = [
    "Model",
    "ModelConfig",
    "DFlashDraftModel",
    "DFlashKVCache",
]
