"""RT-DETRv2 object detection model."""

# Side-effect: ensure the processor module is imported alongside the package.
from . import processing_rt_detr_v2  # noqa: F401

# Sub-config classes used by `ModelConfig.__post_init__` and the framework aliases.
from .config import ModelConfig, RTDetrResNetConfig, RTDetrV2TransformerConfig

# Re-exports: mlx-vlm's framework loader accesses these as attributes on
# the imported package (e.g. `arch.Model`, `arch.LanguageModel`).
from .language import LanguageModel  # noqa: F401
from .rt_detr_v2 import Model  # noqa: F401
from .vision import VisionModel  # noqa: F401

# Aliases for mlx-vlm framework compatibility (update_module_configs).
TextConfig = RTDetrV2TransformerConfig
VisionConfig = RTDetrResNetConfig
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig
