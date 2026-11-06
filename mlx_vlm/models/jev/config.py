from dataclasses import dataclass, field
from typing import Dict

from ..qwen3_5.config import ModelConfig as Qwen3_5ModelConfig


@dataclass
class ModelConfig(Qwen3_5ModelConfig):
    decision_config: Dict = field(default_factory=dict)
