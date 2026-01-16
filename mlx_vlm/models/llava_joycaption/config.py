from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig

# From https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py
# Updated with defaults from https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/blob/main/config.json
 
@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "llama"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    head_dim: int = 128
    max_position_embeddings: int = 131072
    num_key_value_heads: int = 8
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

# From https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/models/siglip.py
# Updated with defaults from https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/blob/main/config.json

@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    layer_norm_eps: float = 1e-6
    vision_use_head: bool = True
    # SigLIP2 parameters
    num_patches: Optional[int] = None  # For SigLIP2, defaults to 256
    max_num_patches: Optional[int] = None  # For naflex variants


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 128077
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None
