"""DeepSeek-V4.1 model package. Importing it installs the processor patch."""

import mlx_vlm.models.deepseek_v41.processing_deepseek_v41  # noqa: F401

from .config import ModelConfig
from .deepseek_v41 import Model

__all__ = ["Model", "ModelConfig"]
