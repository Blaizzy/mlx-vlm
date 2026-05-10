from ....models.qwen3_5.config import TextConfig
from .config import Qwen3_5MTPConfig as ModelConfig
from .qwen3_5_mtp import Qwen3_5MTPDraftModel
from .qwen3_5_mtp import Qwen3_5MTPDraftModel as Model

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "Qwen3_5MTPDraftModel",
]
