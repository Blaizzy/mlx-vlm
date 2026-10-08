"""MapAnything for MLX: feed-forward metric 3D reconstruction from images and
optional calibration, depth and pose inputs.

See README.md for usage and model repos.
"""

from . import processing_mapanything  # Install processor patch
from .config import ModelConfig
from .mapanything import Model
