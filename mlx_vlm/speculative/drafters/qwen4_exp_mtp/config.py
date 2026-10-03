import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.qwen4_exp.config import TextConfig


@dataclass
class Qwen4ExpMTPConfig(BaseModelConfig):
    model_type: str = "qwen4_exp_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is not None:
            self.tie_word_embeddings = bool(self.text_config.tie_word_embeddings)

    @classmethod
    def from_dict(cls, params: dict) -> "Qwen4ExpMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        mtp_depth = int(text_config.get("mtp_num_hidden_layers", 1) or 1)
        flat.setdefault("block_size", mtp_depth + 1)
        sig = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in sig})

    from_hf_dict = from_dict
