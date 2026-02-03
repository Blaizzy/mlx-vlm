from dataclasses import dataclass
from typing import Dict, List, Optional

from ..glm4v.config import ModelConfig as GLM4VModelConfig
from ..glm4v.config import TextConfig, VisionConfig


@dataclass
class ModelConfig(GLM4VModelConfig):
    """Configuration for GLM-OCR model.
    
    GLM-OCR inherits from GLM-4V architecture but is specialized for OCR tasks
    with additional training on document understanding, layout analysis, and
    structured text extraction.
    
    Reference: https://huggingface.co/THUDM/glm-ocr-0.9b
    """
    model_type: str = "glm_ocr"
    vocab_size: int = 151552  # GLM-OCR uses the same vocab as GLM-4V
    ignore_index: int = -100
    image_token_index: int = 151339  # vision_start_token_id from GLM-4V
    image_token_id: int = 151339
    video_token_index: int = 151340  # Not used in OCR but kept for compatibility
    video_token_id: int = 151340
    vision_start_token_id: int = 151339
    vision_end_token_id: int = 151340
    hidden_size: int = 2048
    pad_token_id: int = 151329  # From GLM-4V text config
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        super().__post_init__()
        # GLM-OCR specific defaults if not already set
        if self.eos_token_id is None:
            self.eos_token_id = [151329, 151336, 151338, 151348]
