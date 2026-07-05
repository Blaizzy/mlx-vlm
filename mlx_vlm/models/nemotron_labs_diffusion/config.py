from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nemotron_labs_diffusion"
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[Union[int, list[int]]] = 11
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    sliding_window: Optional[int] = None
    attn_implementation: str = "sdpa"
    mask_token_id: int = 100
    default_generation_mode: str = "ar"
    default_block_length: Optional[int] = None
    default_diffusion_sampler: str = "native"
    default_diffusion_steps: int = 32
    default_diffusion_threshold: Optional[float] = 0.9
    default_diffusion_min_threshold: Optional[float] = 0.45
    default_diffusion_editing_threshold: Optional[float] = 0.9
    default_diffusion_max_post_steps: int = 16
    default_diffusion_num_to_transfer: int = 1
    default_diffusion_max_transfer_per_step: Optional[int] = None
    default_diffusion_stability_steps: int = 2
    default_diffusion_sampling_scaling_factor: float = 2.0
    dlm_paradigm: str = "bidirectional"
    block_size: int = 32
    dlm_loss_weight: Optional[float] = None
    ar_loss_weight: float = 1.0
    dp_varying_mask_ratio: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        rope_parameters = (
            dict(self.rope_parameters)
            if self.rope_parameters is not None
            else (
                dict(self.rope_scaling)
                if self.rope_scaling is not None
                else {"rope_type": "default", "rope_theta": self.rope_theta}
            )
        )
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_parameters
        self.rope_theta = float(rope_parameters.get("rope_theta", self.rope_theta))
