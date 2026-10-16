from dataclasses import dataclass

from ..qwen3_5_moe.config import TextConfig


@dataclass
class ModelConfig(TextConfig):
    def __post_init__(self):
        if self.rope_parameters:
            self.rope_parameters.setdefault("mrope_section", [11, 11, 10])
        super().__post_init__()
