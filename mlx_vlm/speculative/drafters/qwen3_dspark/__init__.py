"""Compatibility package; new code should import ``drafters.dspark``."""

from ..dspark import (
    DSparkDraftModel,
    Model,
    ModelConfig,
    VanillaMarkov,
    validate_dspark_target,
)

__all__ = [
    "DSparkDraftModel",
    "Model",
    "ModelConfig",
    "VanillaMarkov",
    "validate_dspark_target",
]
