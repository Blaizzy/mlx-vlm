"""DINOv2 for MLX.

Channel-last port of the DINOv2 vision transformer, shared as the backbone of
dense-prediction models (Video Depth Anything, MoGe-3). Only the backbone and
its architecture presets are ported; there is no standalone ``Model`` yet.
"""

from .config import DINOV2_PRESETS
from .dinov2 import DINOv2, DINOv2Encoder
