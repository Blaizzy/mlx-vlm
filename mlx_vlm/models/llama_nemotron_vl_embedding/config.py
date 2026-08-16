from dataclasses import dataclass

from ..base import BaseModelConfig
from ..granite_vision.config import VisionConfig
from ..llama.config import ModelConfig as TextConfig

__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "llama_nemotron_vl_embedding"
    downsample_ratio: float = 0.5
    img_context_token_id: int = 128258
    select_layer: int = -1
    force_image_size: int = 512
    normalize: bool = True
