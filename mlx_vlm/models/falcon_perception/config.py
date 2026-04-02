import inspect
from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):

    model_type: str = "falcon_perception"
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    channel_size: int = 3


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "falcon_perception"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    head_dim: int = 128
    num_key_value_heads: int = 8
    vocab_size: int = 65536
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "falcon_perception"
    vocab_size: int = 65536

    img_id: int = 227
    eos_id: int = 11
    image_cls_token_id: int = 244
    image_reg_1_token_id: int = 245
    image_reg_2_token_id: int = 246
    image_reg_3_token_id: int = 247
    image_reg_4_token_id: int = 248
    img_end_id: int = 230

    coord_token_id: int = 240
    size_token_id: int = 241
    seg_token_id: int = 262

    coord_enc_dim: int = 512
    coord_dec_dim: int = 8192
    coord_out_dim: int = 2048
    size_enc_dim: int = 512
    size_dec_dim: int = 8192
    size_out_dim: int = 2048

    do_segmentation: bool = True
    segm_out_dim: int = 256
    num_segm_layers: int = 3

    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        text_params = {
            "model_type": params.get("model_type", "falcon_perception"),
            "hidden_size": params.get("dim", params.get("hidden_size", 1024)),
            "num_hidden_layers": params.get(
                "n_layers", params.get("num_hidden_layers", 28)
            ),
            "num_attention_heads": params.get(
                "n_heads", params.get("num_attention_heads", 16)
            ),
            "head_dim": params.get("head_dim", 128),
            "num_key_value_heads": params.get(
                "n_kv_heads", params.get("num_key_value_heads", 8)
            ),
            "vocab_size": params.get("vocab_size", 65536),
            "intermediate_size": params.get(
                "ffn_dim", params.get("intermediate_size", 3072)
            ),
            "rms_norm_eps": params.get("norm_eps", params.get("rms_norm_eps", 1e-5)),
            "max_position_embeddings": params.get(
                "max_seq_len", params.get("max_position_embeddings", 8192)
            ),
            "rope_theta": float(params.get("rope_theta", 10000)),
        }
        vision_params = {
            "model_type": "falcon_perception",
            "spatial_patch_size": params.get("spatial_patch_size", 16),
            "temporal_patch_size": params.get("temporal_patch_size", 1),
            "channel_size": params.get("channel_size", 3),
        }

        params["text_config"] = text_params
        params["vision_config"] = vision_params

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
