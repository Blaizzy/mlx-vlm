import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.glm5_next.config import TextConfig as Glm5NextTextConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return Glm5NextTextConfig.from_dict(params)


@dataclass
class Glm5NextMTPConfig(BaseModelConfig):
    model_type: str = "glm5_next_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None and self.text_config is not None:
            depth = int(getattr(self.text_config, "num_nextn_predict_layers", 1))
            self.runtime_block_size = min(self.block_size, depth + 1)

    @classmethod
    def from_dict(cls, params: dict) -> "Glm5NextMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = int(text_config.get("num_nextn_predict_layers", 1))
        flat.setdefault("block_size", depth + 1)
        sig = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in sig})

    from_hf_dict = from_dict
