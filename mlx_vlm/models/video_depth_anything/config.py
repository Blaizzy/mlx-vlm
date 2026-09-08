"""Video Depth Anything configuration."""

from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig
from ..dinov2.config import DINOV2_PRESETS

# DPT head presets per encoder variant, on top of the shared DINOv2 dims.
ENCODER_PRESETS = {
    "vits": {
        **DINOV2_PRESETS["vits14"],
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "intermediate_layer_idx": [2, 5, 8, 11],
    },
    "vitb": {
        **DINOV2_PRESETS["vitb14"],
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "intermediate_layer_idx": [2, 5, 8, 11],
    },
    "vitl": {
        **DINOV2_PRESETS["vitl14"],
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "intermediate_layer_idx": [4, 11, 17, 23],
    },
}


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "video_depth_anything"
    encoder: str = "vitl"  # one of "vits", "vitb", "vitl"
    metric: bool = False  # metric (absolute) depth head

    # DPT head
    features: Optional[int] = None  # head feature width; default from encoder preset
    out_channels: Optional[List[int]] = None  # per-level channels; default from preset
    use_bn: bool = False
    use_clstoken: bool = False

    # Temporal module (AnimateDiff-style motion module)
    num_frames: int = 32  # temporal_max_len for the position table
    pe: str = "ape"  # positional embedding type: "ape" (sinusoidal) or "rope"
    num_attention_heads: int = 8
    num_transformer_block: int = 1
    num_attention_blocks: int = 2
    norm_num_groups: int = 32

    # DINOv2 backbone (defaults resolved from the encoder preset)
    embed_dim: Optional[int] = None
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    ffn: Optional[str] = None  # "mlp" or "swiglu"
    intermediate_layer_idx: Optional[List[int]] = None
    img_size: int = 518
    patch_size: int = 14
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    interpolate_offset: float = 0.1
    interpolate_antialias: bool = False

    def __post_init__(self):
        preset = ENCODER_PRESETS.get(self.encoder)
        if preset is None:
            raise ValueError(
                f"Unknown encoder {self.encoder!r}; expected one of {list(ENCODER_PRESETS)}"
            )
        for key, value in preset.items():
            if getattr(self, key) is None:
                setattr(self, key, list(value) if isinstance(value, list) else value)


# Aliases for mlx-vlm framework compatibility (update_module_configs)
TextConfig = ModelConfig
VisionConfig = ModelConfig
