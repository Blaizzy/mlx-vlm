from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig
from ..pixtral import TextConfig, VisionConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = None
    image_token_id: int = None
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
