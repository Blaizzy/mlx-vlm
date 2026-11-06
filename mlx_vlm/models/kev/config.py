from dataclasses import dataclass
from typing import Optional

from ..qwen3_5_text.config import ModelConfig as Qwen3_5TextConfig


@dataclass
class ModelConfig(Qwen3_5TextConfig):
    pointer_dim: int = 256
    model_path: Optional[str] = None
