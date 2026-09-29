"""MoGe-3 configuration.

Mirrors the ``model_config`` dict stored in the official ``model.pt``
checkpoints. Defaults reproduce the released MoGe-3 ViT-L model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..base import BaseModelConfig
from ..dinov2.config import DINOV2_PRESETS

# MoGe names backbones after the torch.hub entry points (``dinov2_vitl14``).
_BACKBONE_PREFIX = "dinov2_"


@dataclass
class EncoderConfig(BaseModelConfig):
    backbone: str = "dinov2_vitl14"
    intermediate_layers: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    dim_out: int = 1024
    img_size: int = 518
    patch_size: int = 14
    mlp_ratio: float = 4.0
    interpolate_offset: float = 0.1
    layer_norm_eps: float = 1e-6
    # Filled from the backbone preset when None; override to shrink for tests.
    embed_dim: Optional[int] = None
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    ffn: Optional[str] = None

    def __post_init__(self):
        preset = DINOV2_PRESETS.get(self.backbone.removeprefix(_BACKBONE_PREFIX))
        if preset is None:
            raise ValueError(
                f"Unknown backbone {self.backbone!r}; expected one of "
                f"{[_BACKBONE_PREFIX + name for name in DINOV2_PRESETS]}"
            )
        for key, value in preset.items():
            if getattr(self, key) is None:
                setattr(self, key, value)


@dataclass
class ConvStackConfig(BaseModelConfig):
    dim_in: List[Optional[int]] = None
    dim_res_blocks: List[int] = None
    dim_out: Optional[List[Optional[int]]] = None
    resamplers: Union[str, List[str]] = "bilinear"
    dim_times_res_block_hidden: int = 1
    num_res_blocks: Union[int, List[int]] = 1
    res_block_in_norm: str = "layer_norm"
    res_block_hidden_norm: str = "group_norm"
    activation: str = "relu"


@dataclass
class ScaleHeadConfig(BaseModelConfig):
    dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024, 1])


@dataclass
class RefinerConfig(BaseModelConfig):
    in_channels: int = 3
    out_channels: int = 1
    encoder_channels: int = 1026
    model_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    encoder_blocks_per_level: Union[int, List[int]] = 1
    decoder_blocks_per_level: Union[int, List[int]] = 1
    bottleneck_blocks: int = 1
    downsample_factors: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_downsample: int = 16


# Default pyramid shared by the neck and the three decoder heads.
_DEFAULT_DIM_RES_BLOCKS = [1024, 256, 128, 64, 32]
_DEFAULT_RESAMPLERS = ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]


def _default_neck() -> ConvStackConfig:
    return ConvStackConfig(
        dim_in=[1026, 2, 2, 2, 2],
        dim_res_blocks=list(_DEFAULT_DIM_RES_BLOCKS),
        dim_out=None,
        num_res_blocks=[0, 2, 2, 2, 0],
        res_block_in_norm="none",
        res_block_hidden_norm="none",
        resamplers=list(_DEFAULT_RESAMPLERS),
    )


def _default_head(dim_out_last: int) -> ConvStackConfig:
    return ConvStackConfig(
        dim_in=list(_DEFAULT_DIM_RES_BLOCKS),
        dim_res_blocks=list(_DEFAULT_DIM_RES_BLOCKS),
        dim_out=[None, None, None, None, dim_out_last],
        num_res_blocks=[0, 1, 1, 1, 0],
        res_block_in_norm="none",
        res_block_hidden_norm="none",
        resamplers=list(_DEFAULT_RESAMPLERS),
    )


_SUB_CONFIGS = {
    "encoder": EncoderConfig,
    "neck": ConvStackConfig,
    "points_head": ConvStackConfig,
    "mask_head": ConvStackConfig,
    "normal_head": ConvStackConfig,
    "scale_head": ScaleHeadConfig,
    "refiner": RefinerConfig,
}


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "moge3"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    neck: ConvStackConfig = field(default_factory=_default_neck)
    points_head: Optional[ConvStackConfig] = field(
        default_factory=lambda: _default_head(3)
    )
    mask_head: Optional[ConvStackConfig] = field(
        default_factory=lambda: _default_head(1)
    )
    normal_head: Optional[ConvStackConfig] = field(
        default_factory=lambda: _default_head(3)
    )
    scale_head: Optional[ScaleHeadConfig] = field(default_factory=ScaleHeadConfig)
    num_tokens_range: List[int] = field(default_factory=lambda: [1200, 3600])
    refiner: Optional[RefinerConfig] = field(default_factory=RefinerConfig)
    refiner_depth_resolution: int = 256

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        params = dict(params)
        for key, sub_cls in _SUB_CONFIGS.items():
            if isinstance(params.get(key), dict):
                params[key] = sub_cls.from_dict(params[key])
        return super().from_dict(params)
