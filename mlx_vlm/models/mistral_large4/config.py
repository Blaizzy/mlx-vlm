from dataclasses import dataclass, field
from typing import Optional

from ..base import BaseModelConfig
from ..mistral3.config import ModelConfig as Mistral3Config
from ..pixtral import VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mistral_large4_text"
    hidden_size: int = 6144
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    num_hidden_layers: int = 72
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 204800
    max_position_embeddings: int = 1048576
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    n_group: int = 1
    topk_group: int = 1
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    final_logit_softcapping: float = 30.0
    rope_parameters: Optional[dict] = None
    no_rope_layers: Optional[list] = None
    kv_source_layers: Optional[list] = None
    indexer_types: Optional[list] = None
    nvfp4: bool = False
    expert_start: int = 0
    expert_end: Optional[int] = None

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = dict(
                rope_theta=10000.0,
                factor=2048.0,
                original_max_position_embeddings=512,
                beta_fast=32.0,
                beta_slow=1.0,
            )
        if self.no_rope_layers is None:
            self.no_rope_layers = [int((i + 1) % 4 != 0) for i in range(n)]
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(n))
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        if self.expert_end is None:
            self.expert_end = self.n_routed_experts
        if not 0 <= self.expert_start < self.expert_end <= self.n_routed_experts:
            raise ValueError("Invalid expert partition")
        for name in ("no_rope_layers", "kv_source_layers", "indexer_types"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} entries")
        for i, source in enumerate(self.kv_source_layers):
            if not 0 <= source <= i or self.kv_source_layers[source] != source:
                raise ValueError(f"Invalid KV source {source} for layer {i}")
            if self.no_rope_layers[source] != self.no_rope_layers[i]:
                raise ValueError("KV sharing requires the same rotary layout")


@dataclass
class ModelConfig(Mistral3Config):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "mistral_large4"
    image_token_index: int = 10
