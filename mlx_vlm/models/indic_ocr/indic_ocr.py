"""IndicOCR OCR-stage model (IndicBlockOCR).

Thin wrapper over the Qwen3.5 implementation: same architecture, same
weight names (``model.language_model.*`` / ``model.visual.*``), Indic
vocab (262157) and token ids carried by ``ModelConfig``.
"""

from dataclasses import replace

from ..qwen3_5.qwen3_5 import LanguageModel
from ..qwen3_5.qwen3_5 import Model as _Qwen35Model
from ..qwen3_5.qwen3_5 import (
    VisionModel,
    sanitize_key,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig, ParserConfig

__all__ = [
    "LanguageModel",
    "Model",
    "VisionModel",
    "sanitize_key",
    "should_shift_norm_weights",
    "should_offset_norm_weight",
]


class Model(_Qwen35Model):
    def __new__(cls, config):
        if isinstance(config, ParserConfig):
            from .pipeline import IndicOCRParser

            return IndicOCRParser.from_config(config)
        return super().__new__(cls)

    def __init__(self, config: ModelConfig):
        # This stage uses Qwen3.5's image-first chat format. Keep the checkpoint
        # model_type for dispatch, but expose the architecture to shared prompting.
        super().__init__(replace(config, model_type="qwen3_5"))
