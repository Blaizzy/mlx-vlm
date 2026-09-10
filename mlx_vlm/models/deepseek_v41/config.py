from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20
    attention_bias: bool = False
    attention_dropout: float = 0.0
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    kv_source_layer_ids: List[int] = field(default_factory=lambda: [2, 8, 14, 20])
    index_source_layer_ids: List[int] = field(
        default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36]
    )
    index_source_layer_ids: List[int] = field(
        default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36]
    )
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: List[int] = field(default_factory=lambda: [1, 14])
    engram_num_embeddings: List[int] = field(
        default_factory=lambda: [384006168, 384016682]
    )
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: List[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    image_token_id: int = 129264
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    vision_num_layers: int = 32
    vision_hidden_size: int = 1024
    vision_num_heads: int = 16
    vision_intermediate_size: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3

    def __post_init__(self):
        if not self.compress_ratios:
            self.compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
        n = self.num_hidden_layers
        if len(self.compress_ratios) == n:
            self.compress_ratios = list(self.compress_ratios) + [0] * (
                self.num_nextn_predict_layers
            )
        if len(self.compress_ratios) != n + self.num_nextn_predict_layers:
            raise ValueError(
                "`compress_ratios` must cover the backbone plus the MTP stages, "
                f"got {len(self.compress_ratios)} for {n} + "
                f"{self.num_nextn_predict_layers} layers."
            )
