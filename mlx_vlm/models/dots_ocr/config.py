import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "dots_ocr"
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = 2
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = True
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_cache: bool = True

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
class VisionConfig(BaseModelConfig):
    model_type: str = "dots_vit"
    embed_dim: int = 1536
    hidden_size: int = 1536
    intermediate_size: int = 4224
    num_hidden_layers: int = 42
    num_attention_heads: int = 12
    num_channels: int = 3
    patch_size: int = 14
    post_norm: bool = True
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    use_bias: bool = False
    attn_implementation: str = "flash_attention_2"
    init_merger_std: float = 0.02
    initializer_range: float = 0.02
    is_causal: bool = False
    gradient_checkpointing: bool = False


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "dots_ocr"
    ignore_index: int = -100
    image_token_id: int = 151665
    video_token_id: int = 151656
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None
    quantization: Optional[Dict] = None

    @classmethod
    def from_dict(cls, params):
        if not params.get("text_config", {}):
            excluded_keys = {"vision_config"}
            params["text_config"] = dict(
                filter(lambda x: x[0] not in excluded_keys, params.items())
            )

        if not params.get("vision_config", {}).get("model_type", {}):
            params["vision_config"]["model_type"] = "dots_vit"

        return cls(
            text_config=TextConfig.from_dict(params["text_config"]),
            vision_config=VisionConfig.from_dict(params["vision_config"]),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in ["text_config", "vision_config"]
            },
        )
