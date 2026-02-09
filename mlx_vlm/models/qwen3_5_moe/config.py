import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig


@dataclass
class VisionConfig(Qwen3VLVisionConfig):
    model_type: str = "qwen3_5_moe"

    def __post_init__(self):
        if (
            self.deepstack_visual_indexes is not None
            and len(self.deepstack_visual_indexes) > 0
        ):
            raise ValueError(
                f"deepstack is disabled for qwen3.5 temporally, but it is set to {self.deepstack_visual_indexes}"
            )
        self.deepstack_visual_indexes = []


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )
    full_attention_interval: int = 4

    def __post_init__(self):
        if self.rope_parameters:
            # Normalize rope_parameters keys (accept both 'rope_type' and 'type')
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            required_keys = {
                "mrope_section",
                "type",
                "rope_theta",
                "partial_rotary_factor",
            }
            if not all(key in self.rope_parameters for key in required_keys):
                raise ValueError(f"rope_parameters must contain keys {required_keys}")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248045
    vision_end_token_id: int = 248046
    vocab_size: int = 248320
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
