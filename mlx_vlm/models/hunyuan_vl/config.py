import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "hunyuan_vl"
    hidden_size: int = 1152
    out_hidden_size: int = 1024
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    num_channels: int = 3
    spatial_merge_size: int = 2
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    interpolate_mode: str = "bilinear"
    cat_extra_token: int = 1
    img_max_token_num: int = 4096
    max_vit_seq_len: int = 16384
    add_patchemb_bias: bool = True
    max_image_size: int = 2048
    hidden_act: str = "gelu"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "hunyuan_vl"
    vocab_size: int = 120818
    org_vocab_size: int = 120818
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 8
    head_dim: Optional[int] = 128
    attention_head_dim: Optional[int] = 128
    intermediate_size: int = 3584
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, int, bool, List[int]]]] = field(
        default_factory=lambda: {
            "alpha": 1000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "type": "xdrope",
            "xdrope_section": [16, 16, 16, 16],
        }
    )
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    norm_type: str = "rms"
    tie_word_embeddings: bool = True
    use_cache: bool = True
    initializer_range: float = 0.02
    routed_scaling_factor: float = 1.0
    dtype: str = "bfloat16"
    bos_token_id: int = 120000
    eos_token_id: int = 120020
    eod_token_id: int = 120020
    pad_token_id: int = -1
    pad_id: int = 120002
    sep_token_id: int = 0
    text_start_id: int = 7
    text_end_id: int = 8
    num_experts: int = 1
    pretraining_tp: int = 1
    use_cla: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.attention_head_dim is None:
            self.attention_head_dim = self.head_dim


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "hunyuan_vl"
    image_start_token_id: int = 120118
    image_end_token_id: int = 120119
    image_token_id: int = 120120
    image_newline_token_id: int = 120121
    bos_token_id: int = 120000
    eos_token_id: int = 120020
    pad_token_id: int = -1
    pad_id: int = 120002
    sep_token_id: int = 0
    text_start_id: int = 7
    text_end_id: int = 8
    vocab_size: int = 120818
    org_vocab_size: int = 120818
    routed_scaling_factor: float = 1.0
    norm_type: str = "rms"
    dtype: str = "bfloat16"
    use_cache: bool = True
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        text_params = params.get("text_config", {})
        vision_params = params.get("vision_config", {})

        for key, value in params.items():
            if key in TextConfig.__dataclass_fields__ and key not in text_params:
                text_params[key] = value
            if key in VisionConfig.__dataclass_fields__ and key not in vision_params:
                vision_params[key] = value

        return cls(
            text_config=TextConfig.from_dict(text_params),
            vision_config=VisionConfig.from_dict(vision_params),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in ["text_config", "vision_config"]
            },
        )
