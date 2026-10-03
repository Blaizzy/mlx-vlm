"""Video Depth Anything for MLX.

Consistent monocular depth estimation for arbitrarily long videos.
Unlike standard VLMs, this model outputs per-frame depth maps.

Usage:
    from mlx_vlm.models.video_depth_anything.generate import VideoDepthPredictor
"""

from . import processing_video_depth_anything  # Install processor patch
from .config import ModelConfig
from .video_depth_anything import Model

# Aliases for mlx-vlm framework compatibility (update_module_configs)
TextConfig = ModelConfig
VisionConfig = ModelConfig
