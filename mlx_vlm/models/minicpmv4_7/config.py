from dataclasses import dataclass
from typing import Optional

from ..minicpmv4_6.config import ModelConfig as MiniCPMVConfig


@dataclass
class ModelConfig(MiniCPMVConfig):
    model_type: str = "minicpmv4_7"
    mrope_mode: str = "canvas"  # Spatial positions for thumbnails and slices.
    image_token_id: Optional[int] = None  # Image feature placeholder.
    video_token_id: Optional[int] = None  # Video feature placeholder.
    image_start_id: Optional[int] = None  # Thumbnail opening marker.
    image_end_id: Optional[int] = None  # Thumbnail closing marker.
    slice_start_id: Optional[int] = None  # Detail crop opening marker.
    slice_end_id: Optional[int] = None  # Detail crop closing marker.
    newline_id: Optional[int] = None  # Separator between rows of slices.

    def __post_init__(self):
        super().__post_init__()
        if self.mrope_mode != "canvas":
            raise ValueError("MiniCPM-V 4.7 requires mrope_mode='canvas'.")
        if self.downsample_mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
