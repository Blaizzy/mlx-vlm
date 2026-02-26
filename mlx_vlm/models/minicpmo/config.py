import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    rope_theta: float
    max_position_embeddings: int

    model_type: str = "minicpmo_text"
    num_experts: int = 0
    num_experts_per_tok: int = 1
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: Optional[int] = None
    norm_topk_prob: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default"}
    )
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    layer_norm_eps: float = 1e-6
    image_size: int = 980
    patch_size: int = 14
    num_channels: int = 3
    spatial_merge_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig

    model_type: str = "minicpmo"
    bos_token_id: int = 151643
    eos_token_id: List[int] = field(default_factory=lambda: [151645, 151643])
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    query_num: int = 64
    use_image_id: bool = True

    @classmethod
    def from_dict(cls, params):
        params = dict(params)

        text_fields = {
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "rms_norm_eps",
            "vocab_size",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
            "attention_bias",
            "attention_dropout",
            "hidden_act",
            "use_cache",
            "use_sliding_window",
            "sliding_window",
            "initializer_range",
            "rope_scaling",
            "num_experts",
            "num_experts_per_tok",
            "decoder_sparse_step",
            "mlp_only_layers",
            "moe_intermediate_size",
        }
        text_config = TextConfig.from_dict({k: v for k, v in params.items() if k in text_fields})

        raw_vision = dict(params.get("vision_config", {}))
        if "hidden_act" not in raw_vision:
            raw_vision["hidden_act"] = "gelu_pytorch_tanh"
        vision_config = VisionConfig.from_dict(raw_vision)

        image_token_id = params.get("image_token_id", 151655)

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            model_type=params.get("model_type", "minicpmo"),
            bos_token_id=params.get("bos_token_id", 151643),
            eos_token_id=params.get("eos_token_id", [151645, 151643]),
            image_token_id=image_token_id,
            video_token_id=params.get("video_token_id", 151656),
            vision_start_token_id=params.get("vision_start_token_id", 151652),
            query_num=params.get("query_num", 64),
            use_image_id=params.get("use_image_id", True),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k
                not in {
                    "text_config",
                    "vision_config",
                    "model_type",
                    "bos_token_id",
                    "eos_token_id",
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "query_num",
                    "use_image_id",
                }
            },
        )
