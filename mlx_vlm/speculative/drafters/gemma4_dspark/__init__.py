from .config import Gemma4DSparkConfig
from .dspark import Gemma4DSparkDraftModel

Model = Gemma4DSparkDraftModel
ModelConfig = Gemma4DSparkConfig

__all__ = [
    "Gemma4DSparkConfig",
    "Gemma4DSparkDraftModel",
    "Model",
    "ModelConfig",
]
