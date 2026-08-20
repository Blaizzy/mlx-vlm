from .config import Qwen3_5MTPConfig as ModelConfig
from .config import TextConfig
from .qwen3_5_mtp import Qwen3_5MTPDraftModel
from .qwen3_5_mtp import Qwen3_5MTPDraftModel as Model

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "Qwen3_5MTPDraftModel",
]
