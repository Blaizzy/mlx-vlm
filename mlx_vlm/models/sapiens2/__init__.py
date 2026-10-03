"""Sapiens2 for MLX: dense-prediction vision transformers, inference only.

See README.md for usage and model repos.
"""

from . import image  # Install processor patch
from .config import HeadConfig, ModelConfig
from .sapiens2 import Model
