"""VisionModel wrapper for mlx-vlm compatibility."""

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class VisionModel(nn.Module):
    """Wraps DINOv3Backbone as a VisionModel for mlx-vlm."""

    def __init__(self, config: VisionConfig = None):
        super().__init__()
        if config is None:
            config = VisionConfig()
        self.config = config
        self.model_type = config.model_type

    def __call__(self, x: mx.array, output_hidden_states: bool = False):
        raise NotImplementedError(
            "VisionModel is accessed through Model.backbone directly. "
            "SAM 3D Body uses ray-conditioned features, not standalone vision encoding."
        )

    @staticmethod
    def sanitize(weights):
        """Pass through — backbone weights are already handled by Model.sanitize()."""
        return weights
