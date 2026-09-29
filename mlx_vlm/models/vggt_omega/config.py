"""VGGT-Omega configuration.

Defaults reproduce the released VGGT-Omega-1B checkpoints: a DINOv3 ViT-L/16
patch encoder, a 24-layer alternating-attention aggregator, and the camera
and depth heads. The text-aligned checkpoint also sets ``enable_alignment``.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "vggt_omega"
    patch_size: int = 16
    embed_dim: int = 1024
    num_heads: int = 16
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-5
    rope_base: float = 100.0

    # DINOv3 patch encoder (``aggregator.patch_embed``).
    encoder_depth: int = 24
    num_storage_tokens: int = 4

    # Aggregator: frame attention, then global or register-only attention.
    depth: int = 24
    num_register_tokens: int = 16
    register_attention_block_indices: List[int] = field(
        default_factory=lambda: [2, 6, 9, 14, 20]
    )
    # Layers read by the heads; the last one must be ``depth - 1``.
    cached_layer_indices: List[int] = field(default_factory=lambda: [4, 11, 17, 23])

    # Heads (inputs are the concatenated frame + cross-frame tokens).
    enable_camera: bool = True
    enable_depth: bool = True
    enable_alignment: bool = False
    head_num_heads: int = 16
    head_depth: int = 4
    dense_features: int = 256
    dense_out_channels: List[int] = field(
        default_factory=lambda: [256, 512, 1024, 1024]
    )
    # Frames per dense-head pass. The 1/4-resolution convs dominate peak
    # memory; 2 was both the fastest and the smallest in tests on M5 Max
    # (the reference uses 8).
    dense_frames_chunk_size: int = 2
    # Weight dtype of the heads at load time. The reference runs them in
    # float32 (the aggregator in bf16); the cost is small. None keeps the
    # checkpoint dtype.
    head_dtype: Optional[str] = "float32"

    # Training resolution; the processor default.
    image_resolution: int = 512

    def __post_init__(self):
        if self.cached_layer_indices[-1] != self.depth - 1:
            raise ValueError("cached_layer_indices must end with the last layer")
        if self.patch_size % 4 != 0:
            raise ValueError("patch_size must be divisible by 4")
