from .config import DeepseekV41DsparkConfig as ModelConfig
from .config import TextConfig
from .deepseek_v41_dspark import DeepseekV41DsparkDraftModel
from .deepseek_v41_dspark import DeepseekV41DsparkDraftModel as Model

__all__ = ["DeepseekV41DsparkDraftModel", "Model", "ModelConfig", "TextConfig"]
