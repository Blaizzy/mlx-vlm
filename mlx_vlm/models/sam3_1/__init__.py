"""SAM 3.1 (Segment Anything Model 3.1) with Object Multiplex for MLX.

Extends SAM 3 with:
- TriViTDetNeck (3-head FPN: detection, interactive, propagation)
- MultiplexMaskDecoder (16 objects simultaneously)
- Decoupled memory attention with image cross-attention

Usage:
    from mlx_vlm.models.sam3_1.generate import Sam3Predictor, Sam3VideoPredictor
"""

from . import processing_sam3_1  # Install processor patch

from .config import ModelConfig
from .sam3_1 import Model

# Reuse text encoder and vision model wrappers from SAM 3
from ..sam3.text_encoder import LanguageModel
from ..sam3.vision import VisionModel

# Required exports for mlx-vlm compatibility
from .config import TextEncoderConfig as TextConfig
from .config import VisionEncoderConfig as VisionConfig
