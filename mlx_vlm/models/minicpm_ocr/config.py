from dataclasses import dataclass

from ..minicpmv4_6.config import ModelConfig as MiniCPMVConfig


@dataclass
class ModelConfig(MiniCPMVConfig):
    model_type: str = "minicpm_ocr"
    downsample_mode: str = "4x"
    image_token_id: int = 248056
    video_token_id: int = 248057

    def __post_init__(self):
        super().__post_init__()
        if self.downsample_mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
