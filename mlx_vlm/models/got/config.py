import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "got"
    # GOT-OCR vision parameters
    img_size: int = 1024
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    out_chans: int = 256
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: tuple = (2, 5, 8, 11)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = 32768
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    sliding_window: int = 32768
    use_sliding_window: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    im_patch_token: int = 151859
    im_start_token: int = 151857
    im_end_token: int = 151858
    vocab_size: int = 151860
    eos_token_id: Optional[Union[int, List[int]]] = 151643

    @classmethod
    def from_dict(cls, params):
        # The GOT-OCR config is flat.
        # Text params are in the root.
        excluded_keys = {"vision_config"}
        text_params = {
            k: v
            for k, v in params.items()
            if k not in excluded_keys and k in inspect.signature(TextConfig).parameters
        }
        params["text_config"] = text_params

        # Vision params are fixed in GOT-OCR, but we allow override if present
        vision_params = params.get("vision_config", {})
        params["vision_config"] = vision_params

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
