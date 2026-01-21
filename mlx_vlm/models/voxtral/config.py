from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "llama"
    hidden_size: int = 3072
    head_dim: int = 128
    num_hidden_layers: int = 30
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 131072
    num_key_value_heads: int = 8
    rope_theta: float = 100000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "voxtral_encoder"
    hidden_size: int = 1280
    head_dim: int = 64
    num_hidden_layers: int = 32
    intermediate_size: int = 5120
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    layerdrop: float = 0.0
    scale_embedding: bool = False
    vocab_size: int = 51866


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "voxtral_vision_stub"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "voxtral"
    hidden_size: Optional[int] = None
    vocab_size: int = 131072
    audio_token_id: Optional[int] = None
    projector_hidden_act: Optional[str] = None
    eos_token_id: Optional[List[int]] = None
