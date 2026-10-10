from dataclasses import dataclass, field
from typing import Dict

from ..qwen3_5.config import ModelConfig as QwenConfig
from ..qwen3_5.config import TextConfig, VisionConfig

__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(QwenConfig):
    joint_head_config: Dict = field(default_factory=dict)
    decision_quantization: str = "preserve-output"

    def __post_init__(self):
        super().__post_init__()
        if not self.joint_head_config:
            raise ValueError("Clef requires joint_head_config and trained head weights")
        if self.decision_quantization not in ("preserve-output", "backbone"):
            raise ValueError("Unknown decision quantization policy")
