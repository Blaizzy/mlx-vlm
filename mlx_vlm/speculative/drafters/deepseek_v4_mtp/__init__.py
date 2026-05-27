from .config import DeepseekV4MTPConfig as ModelConfig
from .config import TextConfig
from .deepseek_v4_mtp import DeepseekV4MTPDraftModel
from .deepseek_v4_mtp import DeepseekV4MTPDraftModel as Model

__all__ = ["DeepseekV4MTPDraftModel", "Model", "ModelConfig", "TextConfig"]
