import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "molmo"
    image_feature_dropout: float = 0.0
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    image_pooling_2d: str = "attention"
    image_projector: str = "mlp"
    eos_token_id: Optional[List[int]] = None


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "molmo"
    max_position_embeddings: int = 4096
    d_model: int = 3584
    n_heads: int = 28
    n_kv_heads: int = 4
    n_layers: int = 28
    mlp_ratio: int = 4
    max_sequence_length: int = 1024
    act_output_multiplier: int = 0.5
    mlp_hidden_size: int = 37888
    vocab_size: int = 152064
    embedding_size: Optional[int] = 152064
    additional_vocab_size: Optional[int] = None
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = -1
    rope: bool = True
    rope_theta: float = 1000000.0
    weight_tying: bool = False
    rope_full_precision: bool = True
    rope_impl: str = "interleave"
    additional_vocab_size: Optional[int] = 128


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "molmo"
    num_channels: int = 3
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    hidden_size: int = 18944
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 23
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "gelu"
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    d_model: int = 3584
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    vit_layers: Optional[List[int]] = field(default_factory=lambda: [-2, -9])
    image_pooling_2d: str = "attention-meanq"
    image_padding_embed: str = "pad_and_partial_pad"
    intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.image_patch_size * self.image_patch_size * 3

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size

    @property
    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w
