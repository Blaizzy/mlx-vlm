"""SAM 3D Body — MLX port for Apple Silicon.

Monocular 3D body mesh recovery from single images.
Outputs SMPL-compatible body mesh vertices, 3D keypoints, and camera params.

mlx-vlm compatible: exports Model, ModelConfig, VisionModel, LanguageModel.
"""

from . import processing_sam3d_body  # noqa: F401
from .config import ModelConfig, SAM3DConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .model import Model, SAM3DBody
from .vision import VisionModel
