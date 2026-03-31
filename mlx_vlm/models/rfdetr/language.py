"""Stub LanguageModel for mlx-vlm framework compatibility."""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class LanguageModel(nn.Module):
    """RF-DETR has no language model. This stub satisfies framework requirements."""

    def __init__(self, config=None):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return None

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return weights
