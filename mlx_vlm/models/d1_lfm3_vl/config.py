from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig
from ..lfm3_vl.config import ModelConfig as Lfm3VlConfig
from ..lfm3_vl.config import TextConfig, VisionConfig
from .box import EDGE_READS
from .prompt import PROMPTS

TRAINED_SET_SIZE = 128  # the largest group a checkpoint records (`d1_set_size`)


@dataclass
class D1Config(BaseModelConfig):
    """The d1 keys of a checkpoint's config.json. A key left out or set to null
    takes the reference runner's default."""

    d1_attention: Optional[str] = None
    d1_set_size: Optional[int] = None
    d1_codes: Optional[int] = None
    d1_numbers: Optional[bool] = None
    d1_vision_max_pixels: Optional[int] = None
    d1_box_edges: Optional[str] = None
    d1_prompt: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.d1_attention = self.d1_attention or "causal"
        self.d1_set_size = 4 if self.d1_set_size is None else self.d1_set_size
        self.d1_codes = self.d1_codes or 1
        self.d1_numbers = bool(self.d1_numbers)
        self.d1_vision_max_pixels = self.d1_vision_max_pixels or 1024 * 1024
        self.d1_box_edges = self.d1_box_edges or "median"
        self.d1_prompt = self.d1_prompt or 1
        if self.d1_attention != "set":
            raise ValueError(
                f"d1 on MLX reads `set` attention checkpoints, not d1_attention={self.d1_attention!r}"
            )
        if not 2 <= self.d1_set_size <= TRAINED_SET_SIZE:
            raise ValueError(f"d1_set_size must be between 2 and {TRAINED_SET_SIZE}")
        if self.d1_box_edges not in EDGE_READS:
            raise ValueError(
                f"d1_box_edges is one of {EDGE_READS}, not {self.d1_box_edges!r}"
            )
        if self.d1_prompt not in PROMPTS:
            raise ValueError(f"d1_prompt is one of {PROMPTS}, not {self.d1_prompt!r}")


@dataclass
class ModelConfig(D1Config, Lfm3VlConfig):
    pass


__all__ = ["D1Config", "ModelConfig", "TextConfig", "VisionConfig"]
