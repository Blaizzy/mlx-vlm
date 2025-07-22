import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ..base import BaseModelConfig


@dataclass
class ProjectorConfig(BaseModelConfig):
    cls: str
    model_type: str
    params: dict


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 102400
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 4096

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
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    cls: str = None
    params: dict = None

    def __post_init__(self):
        if "high_res_cfg" in self.params:
            self.image_size = self.params["high_res_cfg"]["image_size"]


@dataclass
class MLPConfig(BaseModelConfig):
    hidden_size: int
    intermediate_size: int


@dataclass
class SAMViTCfg:
    image_size: Union[Tuple[int, int], int] = 1024
    width: int = 768
    layers: int = 12
    heads: int = 12
    patch_size: int = 16
    window_size: int = 14
    prompt_embed_dim: int = 256
    global_attn_indexes: Union[List[int], Tuple[int]] = (2, 5, 8, 11)
    downsample_channels: Union[List[int], Tuple[int]] = (512, 1024)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    projector_config: ProjectorConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 100015
    vision_feature_select_strategy: str = "default"
    select_layer: int = -1
    pad_id: int = 100001
    num_image_tokens: int = 576
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        if "aligner_config" in params:
            params["projector_config"] = params["aligner_config"]
            del params["aligner_config"]

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
