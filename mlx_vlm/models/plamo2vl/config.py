import inspect
from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    patch_size: int = 14
    image_size: int = 384
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    image_token_id: int = 100002
    image_feature_size: int = 1152
    image_proj_hidden_size: int = 2048

    @classmethod
    def from_dict(cls, params):
        # Upstream flattens the SigLIP encoder config as vision_encoder_* keys
        mapped = {}
        for k, v in dict(params).items():
            if k.startswith("vision_encoder_"):
                mapped[k[len("vision_encoder_") :]] = v
            else:
                mapped[k] = v
        # Upstream config carries an empty model_type; keep our default
        if not mapped.get("model_type"):
            mapped.pop("model_type", None)
        return cls(
            **{
                k: v
                for k, v in mapped.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "plamo2"
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    # Upstream Plamo2Config defaults to tied embeddings when absent from config.json
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 10485760
    attention_window_size: int = 32768
    full_attention_idx: Optional[List[int]] = None
    rope_theta: float = 1000000.0
    rope_local_theta: float = 1000000.0
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 5632
    vocab_size: int = 100032


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "plamo2vl"
    text_config: Optional[TextConfig] = None
    vision_config: Optional[VisionConfig] = None
    image_token_index: Optional[int] = None
    ignore_index: int = -100
    eos_token_id: Optional[List[int]] = None
