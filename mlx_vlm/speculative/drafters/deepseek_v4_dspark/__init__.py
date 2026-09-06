from .config import DeepseekV4DsparkConfig as ModelConfig
from .config import TextConfig
from .deepseek_v4_dspark import DeepseekV4DsparkDraftModel
from .deepseek_v4_dspark import DeepseekV4DsparkDraftModel as Model

__all__ = ["DeepseekV4DsparkDraftModel", "Model", "ModelConfig", "TextConfig"]
