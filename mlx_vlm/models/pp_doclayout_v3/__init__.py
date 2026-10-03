"""PP-DocLayoutV3 document-layout detector."""

from .config import HGNetV2Config, ModelConfig
from .pp_doclayout_v3 import LayoutModel, Model

# Aliases for mlx-vlm framework compatibility (update_module_configs).
TextConfig = ModelConfig
VisionConfig = ModelConfig
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig

__all__ = [
    "AudioConfig",
    "HGNetV2Config",
    "LayoutModel",
    "Model",
    "ModelConfig",
    "PerceiverConfig",
    "ProjectorConfig",
    "TextConfig",
    "VisionConfig",
]
