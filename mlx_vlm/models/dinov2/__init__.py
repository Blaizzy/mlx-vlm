"""DINOv2 for MLX.

Channel-last port of the DINOv2 vision transformer: a standalone image
encoder (``Model``) for the ``facebook/dinov2-*`` and
``facebook/dinov2-with-registers-*`` checkpoints, plus the shared backbone
used by dense-prediction models (Video Depth Anything, MoGe-3). The
training-time DINO heads are not ported.
"""

from .config import DINOV2_PRESETS, ModelConfig
from .dinov2 import DINOv2, DINOv2Encoder, Model

__all__ = ["DINOV2_PRESETS", "DINOv2", "DINOv2Encoder", "Model", "ModelConfig"]
