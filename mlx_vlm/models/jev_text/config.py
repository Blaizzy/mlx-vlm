from dataclasses import dataclass, field
from typing import Dict

from ..qwen3_5_text.config import ModelConfig as Qwen3_5TextConfig


@dataclass
class ModelConfig(Qwen3_5TextConfig):
    decision_config: Dict = field(default_factory=dict)
