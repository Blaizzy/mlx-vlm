"""GLM-OCR Model wrapper.

GLM-OCR (0.9B) is a multimodal OCR model built on GLM-V architecture.
It inherits from GLM-4V but is specialized for document understanding.
"""

import mlx.core as mx
import mlx.nn as nn

from ..glm4v.glm4v import Model as GLM4VModel
from .config import ModelConfig
from .processing import GlmOcrProcessor


class Model(GLM4VModel):
    """GLM-OCR Model wrapper.
    
    This class wraps the GLM-4V implementation with GLM-OCR specific configuration.
    GLM-OCR uses the same architecture as GLM-4V but with:
    - Smaller size (0.9B parameters vs larger GLM-4V variants)
    - Specialized training on OCR tasks
    - Optimized for document layout understanding
    
    The model inherits all functionality from GLM4VModel but registers with
    the correct model_type for mlx-vlm loading.
    """
    
    def __init__(self, config: ModelConfig):
        # Initialize parent GLM-4V model
        super().__init__(config)
        self.config = config


# Register the processor for the GLM-OCR model
try:
    from transformers import AutoProcessor
    AutoProcessor.register("GlmOcrProcessor", GlmOcrProcessor)
except Exception as e:
    print(f"Warning: Could not register GLM-OCR processor: {e}")
