import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.qwen4_exp.config import TextConfig as Qwen4ExpTextConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return Qwen4ExpTextConfig.from_dict(params)


@dataclass
class Qwen4ExpMTPConfig(BaseModelConfig):
    model_type: str = "qwen4_exp_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None and self.text_config is not None:
            depth = getattr(self.text_config, "mtp_num_hidden_layers", 1)
            self.runtime_block_size = min(self.block_size, int(depth) + 1)

    @classmethod
    def from_dict(cls, params: dict) -> "Qwen4ExpMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        depth = text_config.get("mtp_num_hidden_layers", 1)
        flat.setdefault("block_size", int(depth) + 1)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
