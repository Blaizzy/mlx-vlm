"""Stub LanguageModel for mlx-vlm compatibility.

SAM 3D Body is a vision-only model with no text encoder.
This stub satisfies the mlx-vlm framework's requirement for a LanguageModel export.
"""

import mlx.nn as nn

from .config import TextConfig


class LanguageModel(nn.Module):
    """Stub — SAM 3D Body has no text encoder."""

    def __init__(self, config: TextConfig = None):
        super().__init__()
        self.model_type = "none"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("SAM 3D Body does not use a language model.")
