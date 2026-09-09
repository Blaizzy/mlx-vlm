from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class BackboneConfig(BaseModelConfig):
    model_type: str = "pp_lcnet_v4"
    scale: float = 1.0
    stem_channels: List[int] = field(default_factory=lambda: [3, 48, 96])
    stem_type: str = "large"
    stem_strides: List[int] = field(default_factory=lambda: [2, 1, 1, 2, 1])
    block_configs: List[List[List[Any]]] = field(default_factory=list)
    out_features: Optional[List[str]] = None
    out_indices: Optional[List[int]] = None
    reduction: int = 4
    hidden_act: str = "relu"
    num_channels: int = 3
    use_learnable_affine_block: bool = False

    @property
    def stage_out_channels(self):
        return [blocks[-1][2] for blocks in self.block_configs]


@dataclass
class ModelConfig(BaseModelConfig):
    backbone_config: BackboneConfig
    model_type: str = "pp_ocrv6_small_rec"

    # Detection
    reduction: int = 4
    layer_list_out_channels: Optional[List[int]] = None
    neck_out_channels: int = 96
    kernel_list: List[int] = field(default_factory=lambda: [3, 2, 2])
    interpolate_mode: str = "nearest"
    dilated_kernel_size: int = 7
    mode: str = "large"
    scale_factor_list: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    intraclass_block_number: int = 4
    intraclass_block_config: Optional[Dict[str, Any]] = None
    reduce_factor: int = 2

    # Recognition
    hidden_act: str = "silu"
    hidden_size: int = 120
    mlp_ratio: float = 2.0
    depth: int = 2
    head_out_channels: int = 18710
    conv_kernel_size: List[int] = field(default_factory=lambda: [1, 7])
    qkv_bias: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        backbone = params.get("backbone_config", {})
        if isinstance(backbone, BackboneConfig):
            backbone_config = backbone
        else:
            backbone_config = BackboneConfig.from_dict(backbone)
        params["backbone_config"] = backbone_config
        return super().from_dict(params)

    @property
    def is_detection(self):
        return self.model_type.endswith("_det")

    @property
    def is_recognition(self):
        return self.model_type.endswith("_rec")
