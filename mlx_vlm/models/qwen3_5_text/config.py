from dataclasses import dataclass
from typing import Dict, Optional

from ..qwen3_5.config import TextConfig, sanitize_quantization_config


@dataclass
class ModelConfig(TextConfig):
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        super().__post_init__()
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )
