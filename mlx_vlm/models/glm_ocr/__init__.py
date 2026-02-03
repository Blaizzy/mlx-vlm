"""GLM-OCR model support for mlx-vlm.

GLM-OCR is a multimodal OCR model for complex document understanding,
built on the GLM-V encoder-decoder architecture.

Model: zai-org/GLM-OCR
"""

from .config import ModelConfig, TextConfig, VisionConfig
from .model import LanguageModel, Model, VisionModel
from .processing import Glm46VProcessor

__all__ = [
    "Model",
    "ModelConfig", 
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
    "Glm46VProcessor",
]
