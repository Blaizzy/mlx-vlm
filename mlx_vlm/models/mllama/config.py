from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mllama"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    cross_attention_layers: List[int] = field(
        default_factory=lambda: [3, 8, 13, 18, 23, 28, 33, 38]
    )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    image_size: int = 560
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    max_num_tiles: int = 4
    max_aspect_ratio_id: int = 8
    num_global_layers: int = 8
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    vision_output_dim: int = 7680
    intermediate_layers_indices: List[int] = field(
        default_factory=lambda: [3, 7, 15, 23, 30]
    )
    supported_aspect_ratios: Tuple[List[int]] = (
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 1],
        [2, 2],
        [3, 1],
        [4, 1],
    )


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 128256
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None
