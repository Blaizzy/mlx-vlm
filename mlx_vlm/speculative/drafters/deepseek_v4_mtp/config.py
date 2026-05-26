import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return DeepseekV4Config.from_dict(params)


@dataclass
class DeepseekV4MTPConfig(BaseModelConfig):
    model_type: str = "deepseek_v4_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 3
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV4MTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        nextn_depth = text_config.get("num_nextn_predict_layers", 1)
        flat.setdefault("block_size", int(nextn_depth) + 2)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
