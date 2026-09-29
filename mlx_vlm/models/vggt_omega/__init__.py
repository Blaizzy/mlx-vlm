"""VGGT-Omega for MLX: feed-forward camera and depth reconstruction.

See README.md for usage and model repos.
"""

from . import processing_vggt_omega  # Install processor patch
from .config import ModelConfig
from .vggt_omega import Model
