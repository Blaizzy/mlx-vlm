"""RT-DETRv2 object detection model."""

# Side effect: registers RTDetrV2Processor with transformers.AutoProcessor
# via install_auto_processor_patch (see processing_rt_detr_v2.py).
from . import processing_rt_detr_v2
from .config import ModelConfig, RTDetrResNetConfig, RTDetrV2TransformerConfig

# Re-exports: mlx-vlm's framework loader accesses these as attributes on
# the imported package (e.g. `arch.Model`, `arch.LanguageModel`).
from .language import LanguageModel
from .rt_detr_v2 import Model
from .vision import VisionModel

# Aliases for mlx-vlm framework compatibility (update_module_configs).
TextConfig = RTDetrV2TransformerConfig
VisionConfig = RTDetrResNetConfig
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig

__all__ = [
    "AudioConfig",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "PerceiverConfig",
    "ProjectorConfig",
    "RTDetrResNetConfig",
    "RTDetrV2TransformerConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
    "processing_rt_detr_v2",
]
