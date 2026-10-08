from dataclasses import dataclass, field
from typing import Tuple

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """YOLO11 detector geometry, as written by ``convert.py``."""

    model_type: str = "yolo11"
    nc: int = 1  # number of classes
    ch: Tuple[int, int, int] = field(default_factory=lambda: (256, 512, 512))
    reg_max: int = 16  # DFL bins per box side
