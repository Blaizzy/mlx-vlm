import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "paligemma"
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        text_params = params.get("text_config", {})
        vision_params = params.get("vision_config", {})

        for key, value in params.items():
            if key in TextConfig.__dataclass_fields__ and key not in text_params:
                text_params[key] = value
            if key in VisionConfig.__dataclass_fields__ and key not in vision_params:
                vision_params[key] = value

        # HF compatibility: PaliGemma defaults to bidirectional text attention
        # when the field is unset in config.
        if text_params.get("use_bidirectional_attention", None) is None:
            text_params["use_bidirectional_attention"] = True

        # Older checkpoints may only provide hidden_act.
        if "hidden_activation" not in text_params and "hidden_act" in text_params:
            text_params["hidden_activation"] = text_params["hidden_act"]

        if "projection_dim" in params and "projection_dim" not in vision_params:
            vision_params["projection_dim"] = params["projection_dim"]

        image_size = vision_params.get("image_size")
        patch_size = vision_params.get("patch_size")
        if (
            image_size is not None
            and patch_size is not None
            and "num_image_tokens" not in text_params
        ):
            text_params["num_image_tokens"] = (image_size // patch_size) ** 2

        return cls(
            text_config=TextConfig.from_dict(text_params),
            vision_config=VisionConfig.from_dict(vision_params),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in {"text_config", "vision_config"}
            },
        )


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "gemma"
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 16
    vocab_size: int = 256000
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    max_position_embeddings: int = 4096
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 4096
    layer_types: Optional[List[str]] = None
    num_image_tokens: Optional[int] = None
    use_bidirectional_attention: Optional[bool] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None and self.model_type == "gemma2":
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    patch_size: int = 14
    projection_dim: int = 2048
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
