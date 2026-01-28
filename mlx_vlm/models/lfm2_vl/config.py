from dataclasses import dataclass
from typing import List, Tuple

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "lfm2"
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    intermediate_size: int = 6656
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    vocab_size: int = 65536
    eos_token_id: int = 7
    initializer_range: float = 0.02
    norm_eps: float = 1e-05
    use_cache: bool = True
    use_pos_enc: bool = True
    block_auto_adjust_ff_dim: bool = True
    block_dim: int = 1024
    block_ff_dim: int = 6656
    block_ffn_dim_multiplier: float = 1.0
    block_mlp_init_scale: float = 1.0
    block_multiple_of: int = 256
    block_norm_eps: float = 1e-05
    block_out_init_scale: float = 1.0
    block_use_swiglu: bool = True
    block_use_xavier_init: bool = True
    conv_L_cache: int = 3
    conv_bias: bool = False
    conv_dim: int = 1024
    conv_dim_out: int = 1024
    conv_use_xavier_init: bool = True
    layer_types: List[str] = None
    num_heads: int = 16
    full_attn_idxs: List[int] = None

    def __post_init__(self):

        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "lfm2_vl"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 256
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-06
    hidden_act: str = "gelu_pytorch_tanh"
    vision_use_head: bool = False
    num_positions: int = None
    spatial_shapes: List[Tuple[int, int]] = None


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "lfm2-vl"
    do_image_splitting: bool = True
    downsample_factor: int = 2
    encoder_patch_size: int = 16
    image_token_index: int = 396
    max_image_tokens: int = 256
    max_num_patches: int = 1024
    max_pixels_tolerance: float = 2.0
    max_tiles: int = 10
    min_image_tokens: int = 64
    min_tiles: int = 2
    tile_size: int = 512
    use_image_special_tokens: bool = True
    use_thumbnail: bool = False
    vision_feature_layer: int = -1
    projector_bias: bool = True
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 2560
    eos_token_id: int = 7
    projector_use_layernorm: bool = True
