from dataclasses import dataclass, field
from typing import Dict

from ..qwen3_5.config import ModelConfig as Qwen3_5ModelConfig


@dataclass
class ModelConfig(Qwen3_5ModelConfig):
    head_config: Dict = field(default_factory=dict)
    decision_quantization: str = "preserve-output"

    def __post_init__(self):
        super().__post_init__()
        if self.decision_quantization not in ("preserve-output", "backbone"):
            raise ValueError("Unknown decision quantization policy")
