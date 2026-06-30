from .config import DeepseekV4DSparkConfig
from .dspark import DeepseekV4DSparkDraftModel

Model = DeepseekV4DSparkDraftModel
ModelConfig = DeepseekV4DSparkConfig

__all__ = [
    "DeepseekV4DSparkConfig",
    "DeepseekV4DSparkDraftModel",
    "Model",
    "ModelConfig",
]
