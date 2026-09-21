import mlx.nn as nn

from .config import VisionConfig


class VisionModel(nn.Module):
    """MiMo-V2.6 vision tower (``vision_model_type: mimovl``).

    Not yet ported. The reference tower is a 28-layer ViT whose blocks alternate
    between full, windowed and strided attention per ``vit_window_attn_types``,
    with an attention sink and grouped-query projections, followed by a
    ``spatial_merge_size`` patch merger into ``out_hidden_size``.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("MiMo-V2.6 vision tower is not implemented yet")

    def sanitize(self, weights):
        return weights
