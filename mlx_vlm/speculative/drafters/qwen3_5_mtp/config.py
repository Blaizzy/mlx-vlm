import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.qwen3_5.config import TextConfig


@dataclass
class Qwen3_5MTPConfig(BaseModelConfig):
    model_type: str = "qwen3_5_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 3
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            raw_text_config = self.text_config
            self.text_config = TextConfig.from_dict(raw_text_config)
            for key in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"):
                if key in raw_text_config:
                    setattr(self.text_config, key, raw_text_config[key])
        if self.text_config is not None:
            self.tie_word_embeddings = bool(self.text_config.tie_word_embeddings)

    @classmethod
    def from_dict(cls, params: dict) -> "Qwen3_5MTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        mtp_depth = text_config.get("mtp_num_hidden_layers", 1)
        flat.setdefault("block_size", int(mtp_depth) + 2)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
