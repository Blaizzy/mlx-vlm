"""Backward-compatible imports for the model-agnostic DSpark runtime."""

from ..dspark.dspark import (
    DSparkConfidenceHead,
    DSparkDraftModel,
    VanillaMarkov,
    validate_dspark_target,
)

Model = DSparkDraftModel

__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftModel",
    "Model",
    "VanillaMarkov",
    "validate_dspark_target",
]
