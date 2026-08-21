"""Compatibility package; new code should import ``drafters.dspark``."""

from ..dspark import DSparkDraftModel, Model, ModelConfig, VanillaMarkov
from .dspark import (
    validate_dspark_target,
    validate_lfm2_dspark_target,
    validate_qwen3_5_dspark_target,
)

__all__ = [
    "DSparkDraftModel",
    "Model",
    "ModelConfig",
    "VanillaMarkov",
    "validate_dspark_target",
    "validate_lfm2_dspark_target",
    "validate_qwen3_5_dspark_target",
]
