from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "moonvit"
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    num_channels: int = 3
    merge_kernel_size: List[int] = field(default_factory=lambda: [2, 2])

    def __post_init__(self):
        if self.merge_kernel_size is None:
            self.merge_kernel_size = [2, 2]
        self.depth = self.num_hidden_layers
        self.num_heads = self.num_attention_heads
        self.embed_dim = self.hidden_size
        self.spatial_merge_size = self.merge_kernel_size[0]


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "qwen2"
    hidden_size: int = 2048
    num_hidden_layers: int = 36
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 2
    vocab_size: int = 152681
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    block_size: int = 6
    causal_attn: bool = False
    text_mask_token_id: int = 151676
    null_token_id: int = 152678
    switch_token_id: int = 152679

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "locateanything"
    image_token_index: int = 151665
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    none_token_id: int = 4064
    mlp_connector_layers: int = 2
    vocab_size: int = 152681
    eos_token_id: Optional[List[int]] = None
    n_future_tokens: int = 6
