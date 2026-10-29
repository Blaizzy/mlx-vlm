import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3.config import ModelConfig as TextConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip2_navit"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    image_size: int = 512
    hidden_stride: int = 2
    temporal_patch_size: int = 1
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "oxr"
    hidden_size: int = 1024
    visual_vocab_size: int = 65536
    image_token_id: int = 151655  # <|image_pad|>
    vision_start_token_id: int = 151652  # <|vision_start|>
    vision_end_token_id: int = 151653  # <|vision_end|>
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        # The OXR checkpoint names its sub-configs llm_config / vit_config.
        if "text_config" not in params and "llm_config" in params:
            params["text_config"] = params["llm_config"]
        if "vision_config" not in params and "vit_config" in params:
            params["vision_config"] = params["vit_config"]
        text = params["text_config"]
        vision = params["vision_config"]
        return cls(
            text_config=(
                text if isinstance(text, TextConfig) else TextConfig.from_dict(text)
            ),
            vision_config=(
                vision
                if isinstance(vision, VisionConfig)
                else VisionConfig.from_dict(vision)
            ),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in ("text_config", "vision_config")
            },
        )
