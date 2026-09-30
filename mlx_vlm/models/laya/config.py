from dataclasses import dataclass, field
from typing import Dict

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "laya"
    encoder_config: Dict = field(default_factory=dict)
    decision_config: Dict = field(default_factory=dict)
    head_layers: int = 2
    tokenizer_subfolder: str = "tokenizer"

    def __post_init__(self):
        self.encoder_config = dict(self.encoder_config)
        rope = self.encoder_config.get("rope_parameters", {})
        if isinstance(rope, dict):
            for kind, key, default in (
                ("full_attention", "global_rope_theta", 160000),
                ("sliding_attention", "local_rope_theta", 10000),
            ):
                self.encoder_config[key] = rope.get(kind, {}).get(
                    "rope_theta", self.encoder_config.get(key, default)
                )
