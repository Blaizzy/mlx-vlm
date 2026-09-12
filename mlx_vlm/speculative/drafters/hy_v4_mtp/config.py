import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.hy_v4.config import ModelConfig as HyV4Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return HyV4Config.from_dict(params)


@dataclass
class HyV4MTPConfig(BaseModelConfig):
    model_type: str = "hy_v4_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None:
            self.runtime_block_size = min(self.block_size, 2)

    @classmethod
    def from_dict(cls, params: dict) -> "HyV4MTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = int(text_config.get("num_nextn_predict_layers", 1) or 1)
        flat.setdefault("block_size", depth + 1)
        flat.setdefault("runtime_block_size", min(int(flat["block_size"]), depth + 1))
        sig = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in sig})

    from_hf_dict = from_dict
