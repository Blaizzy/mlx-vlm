from dataclasses import dataclass, field
from typing import Dict

from ..qwen3_5_text.config import ModelConfig as QwenConfig


@dataclass
class ModelConfig(QwenConfig):
    decision_config: Dict = field(default_factory=dict)
