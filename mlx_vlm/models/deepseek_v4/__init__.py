import mlx_vlm.models.deepseek_v4.processing_deepseek_v4  # noqa: F401 (installs processor patch)

from .config import ModelConfig
from .deepseek_v4 import Model

__all__ = ["Model", "ModelConfig"]
