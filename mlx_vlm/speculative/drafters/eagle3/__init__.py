from .config import Eagle3Config as ModelConfig
from .config import TextConfig
from .eagle3 import Eagle3DraftModel
from .eagle3 import Eagle3DraftModel as Model

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "Eagle3DraftModel",
]
