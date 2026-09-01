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
    text_config: Optional[object] = None
    block_size: int = 3
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = Glm5NextTextConfig.from_dict(self.text_config)
        if self.text_config is not None:
            self.tie_word_embeddings = bool(
                getattr(self.text_config, "tie_word_embeddings", False)
            )

    @classmethod
    def from_dict(cls, params: dict) -> "Glm5NextMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = text_config.get("num_nextn_predict_layers", 1)
        flat["block_size"] = max(int(flat.get("block_size", 0)), int(depth) + 2)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
