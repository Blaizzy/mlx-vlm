import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.longcat_flash_sparse.config import ModelConfig as LongcatSparseConfig


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return LongcatSparseConfig.from_dict(params)


@dataclass
class LongcatFlashSparseMTPConfig(BaseModelConfig):
    model_type: str = "longcat_flash_sparse_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 4  # mtp_num_layers + 1
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: dict) -> "LongcatFlashSparseMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        mtp_depth = text_config.get("mtp_num_layers", 3)
        flat.setdefault("block_size", int(mtp_depth) + 1)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
