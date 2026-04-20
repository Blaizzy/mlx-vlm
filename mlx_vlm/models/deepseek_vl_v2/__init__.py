import mlx_vlm.models.deepseek_vl_v2.processing_deepsek_vl_v2  # noqa: F401 (installs processor patch)

from .config import MLPConfig, ModelConfig, ProjectorConfig, TextConfig, VisionConfig
from .deepseek_vl_v2 import DeepseekVLV2Processor, LanguageModel, Model, VisionModel
