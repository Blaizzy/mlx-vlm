from .config import ModelConfig, TextConfig
from .deepseek_v41_dspark import DeepseekV41DsparkDraftModel as Model

__all__ = ["Model", "ModelConfig", "TextConfig"]
