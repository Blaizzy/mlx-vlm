from . import processing_rfdetr  # Install processor patch (side effect)
from .config import DINOv2Config, ModelConfig, TransformerConfig
from .language import LanguageModel
from .rfdetr import Model
from .vision import VisionModel

# Aliases for mlx-vlm framework compatibility (update_module_configs)
TextConfig = TransformerConfig
VisionConfig = DINOv2Config
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig
