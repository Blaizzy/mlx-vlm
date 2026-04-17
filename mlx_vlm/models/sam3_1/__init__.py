"""SAM 3.1 (Segment Anything Model 3.1) with Object Multiplex for MLX.

Extends SAM 3 with:
- TriViTDetNeck (3-head FPN: detection, interactive, propagation)
- MultiplexMaskDecoder (16 objects simultaneously)
- Decoupled memory attention with image cross-attention

Usage:
    from mlx_vlm.models.sam3_1.generate import Sam3Predictor, Sam3VideoPredictor
"""

from ..sam3.text_encoder import LanguageModel  # noqa: F401
from ..sam3.vision import VisionModel  # noqa: F401
from . import processing_sam3_1  # noqa: F401

# Required exports for mlx-vlm compatibility
from .config import ModelConfig
from .config import TextEncoderConfig as TextConfig
from .config import VisionEncoderConfig as VisionConfig
from .sam3_1 import Model
