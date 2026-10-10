import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.glm_moe_dsa.config import ModelConfig as GlmMoeDsaTextConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return GlmMoeDsaTextConfig.from_dict(params)


@dataclass
class GlmMoeDsaMTPConfig(BaseModelConfig):
    model_type: str = "glm_moe_dsa_mtp"
    text_config: Optional[object] = None
    block_size: int = 2
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is not None:
            self.tie_word_embeddings = bool(
                getattr(self.text_config, "tie_word_embeddings", False)
            )

    @classmethod
    def from_dict(cls, params: dict) -> "GlmMoeDsaMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = text_config.get("num_nextn_predict_layers", 1)
        flat.setdefault("block_size", int(depth) + 1)
        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict
