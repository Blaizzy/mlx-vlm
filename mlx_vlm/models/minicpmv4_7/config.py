from dataclasses import dataclass
from typing import Optional

from ..minicpmv4_6.config import ModelConfig as MiniCPMVConfig
from ..minicpmv4_6.config import TextConfig as MiniCPMVTextConfig
from ..qwen3_5_moe.config import TextConfig as MoETextConfig


class TextConfig(MiniCPMVTextConfig):
    @classmethod
    def from_dict(cls, params):
        if params.get("model_type") == "qwen3_5_moe_text":
            return MoETextConfig.from_dict(params)
        return super().from_dict(params)


@dataclass
class ModelConfig(MiniCPMVConfig):
    text_config: TextConfig | MoETextConfig
    model_type: str = "minicpmv4_7"
    mrope_mode: str = "canvas"  # Spatial positions for thumbnails and slices.
    image_token_id: Optional[int] = None  # Image feature placeholder.
    video_token_id: Optional[int] = None  # Video feature placeholder.
    image_start_id: Optional[int] = None  # Thumbnail opening marker.
    image_end_id: Optional[int] = None  # Thumbnail closing marker.
    slice_start_id: Optional[int] = None  # Detail crop opening marker.
    slice_end_id: Optional[int] = None  # Detail crop closing marker.
    newline_id: Optional[int] = None  # Separator between rows of slices.

    @staticmethod
    def _parse_text_config(params):
        return TextConfig.from_dict(params)

    def __post_init__(self):
        super().__post_init__()
        if self.mrope_mode != "canvas":
            raise ValueError("MiniCPM-V 4.7 requires mrope_mode='canvas'.")
        if self.downsample_mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
