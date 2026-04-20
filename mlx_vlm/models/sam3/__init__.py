"""SAM3 (Segment Anything Model 3) for MLX.

Open-vocabulary detection, segmentation, and tracking model.
Unlike standard VLMs, SAM3 outputs masks/boxes instead of text.

Usage:
    from mlx_vlm.models.sam3.generate import Sam3Predictor, Sam3VideoPredictor
"""

from . import processing_sam3  # Install processor patch
from .config import (
    DetectorConfig,
    DetectorMaskDecoderConfig,
    DETRDecoderConfig,
    DETREncoderConfig,
    GeometryEncoderConfig,
    ModelConfig,
    TextEncoderConfig,
    TrackerConfig,
    TrackerMaskDecoderConfig,
    VisionEncoderConfig,
    ViTConfig,
)
from .sam3 import Model
from .text_encoder import LanguageModel
from .vision import VisionModel

# Required exports for mlx-vlm compatibility
TextConfig = TextEncoderConfig
VisionConfig = VisionEncoderConfig
