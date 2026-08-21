"""Backward-compatible imports for the model-agnostic DSpark runtime."""

from ..dspark.dspark import (
    DSparkConfidenceHead,
    DSparkDraftModel,
    VanillaMarkov,
    validate_dspark_target,
)

# These names shipped with the original LFM-specific module. Keep them as
# import aliases while all validation is handled by the structural contract.
validate_lfm2_dspark_target = validate_dspark_target
validate_qwen3_5_dspark_target = validate_dspark_target
Model = DSparkDraftModel

__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftModel",
    "Model",
    "VanillaMarkov",
    "validate_dspark_target",
    "validate_lfm2_dspark_target",
    "validate_qwen3_5_dspark_target",
]
