"""RTMDet configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..base import BaseModelConfig

# Arch multipliers mirror mmdetection's `arch_settings` for CSPNeXt / RTMDet.
# For the "m" (medium) size: deepen_factor=0.67, widen_factor=0.75.  With base
# channels [64, 128, 256, 512, 1024] and base num_blocks [3, 6, 6, 3], we get:
#   channels = round(base * 0.75) = [48, 96, 192, 384, 768]
#   num_blocks = round(base * 0.67) = [2, 4, 4, 2]
ARCH_ZOO = {
    # name → (channels (len 5), num_blocks (len 4))
    "tiny": ([32, 48, 96, 192, 384], [1, 3, 3, 1]),
    "s":    ([32, 64, 128, 256, 512], [1, 2, 2, 1]),
    "m":    ([48, 96, 192, 384, 768], [2, 4, 4, 2]),
    "l":    ([64, 128, 256, 512, 1024], [3, 6, 6, 3]),
    "x":    ([80, 160, 320, 640, 1280], [4, 8, 8, 4]),
}


@dataclass
class RTMDetConfig(BaseModelConfig):
    model_type: str = "rtmdet"
    arch: str = "m"
    num_classes: int = 1  # person detector ships single-class
    image_size: Tuple[int, int] = (640, 640)  # (H, W)
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    neck_channels: int = 192  # unified channel count after neck (= channels[2] for "m")
    head_stacked_convs: int = 2

    # inference hyperparams
    score_threshold: float = 0.3
    nms_iou_threshold: float = 0.65
    max_detections: int = 300

    def __post_init__(self):
        # mlx-vlm framework compatibility hooks
        self.text_config = None
        self.vision_config = None

    def channels(self) -> List[int]:
        return ARCH_ZOO[self.arch][0]

    def num_blocks(self) -> List[int]:
        return ARCH_ZOO[self.arch][1]
