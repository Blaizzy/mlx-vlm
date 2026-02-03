"""GLM-OCR model support for mlx-vlm.

GLM-OCR is a multimodal OCR model for complex document understanding,
built on the GLM-V encoder-decoder architecture.

Model: THUDM/glm-ocr-0.9b
"""

from ..glm4v.config import TextConfig, VisionConfig
from ..glm4v.glm4v import LanguageModel, VisionModel
from .config import ModelConfig
from .model import Model
from .processing import GlmOcrProcessor

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
    "GlmOcrProcessor",
]
