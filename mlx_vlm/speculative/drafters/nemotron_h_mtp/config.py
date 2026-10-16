from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.nemotron_h.config import ModelConfig as TextConfig


@dataclass
class NemotronHMTPConfig(BaseModelConfig):
    model_type: str = "nemotron_h_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is None:
            raise ValueError("Nemotron MTP requires text_config.")
        if self.text_config.num_nextn_predict_layers != 1:
            raise ValueError("Nemotron MTP requires num_nextn_predict_layers=1.")

    @property
    def layer_pattern(self):
        text = self.text_config
        pattern = text.mtp_hybrid_override_pattern
        if pattern is None and text.mtp_layers_block_type is not None:
            pattern = [text._block_type_to_char[t] for t in text.mtp_layers_block_type]
        if not pattern or any(t not in ("*", "E") for t in pattern):
            raise ValueError("Nemotron MTP requires an attention/MoE layer pattern.")
        return pattern
