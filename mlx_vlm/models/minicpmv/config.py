import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class SliceConfig(BaseModelConfig):
    model_type: str = "minicpmv4_6"
    patch_size: int = 14
    max_slice_nums: int = 9
    scale_resolution: int = 448

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params.pop("model_type", None)
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.model_type == "siglip":
            self.model_type = "siglip_vision_model"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    linear_num_value_heads: int = 16
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict[str, Union[float, str, bool, list[int]]]] = None
    rope_parameters: Optional[Dict[str, Union[float, str, bool, list[int]]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
    full_attention_interval: int = 4

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_parameters is None and self.rope_scaling is not None:
            self.rope_parameters = dict(self.rope_scaling)
        elif self.rope_parameters is None:
            self.rope_parameters = {
                "type": "default",
                "mrope_section": [11, 11, 10],
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": 0.25,
            }

        if "type" not in self.rope_parameters and "rope_type" in self.rope_parameters:
            self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
        self.rope_parameters.setdefault("type", "default")
        self.rope_parameters.setdefault("mrope_section", [11, 11, 10])
        self.rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", 0.25)

        if self.rope_scaling is None:
            self.rope_scaling = dict(self.rope_parameters)
        if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
            self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "minicpmv"
    query_num: int = 64
    image_size: int = 448
    patch_size: int = 14
    init_vision: bool = True
    batch_vision_input: bool = True
    vision_batch_size: int = 16
    stream_input: bool = True
    slice_mode: bool = True
    slice_config: Optional[SliceConfig] = None
    insert_layer_id: int = 6
    downsample_mode: str = "16x"
    merge_kernel_size: tuple[int, int] = (2, 2)
    merger_times: int = 1
    eos_token_id: Optional[list[int]] = None

    def __post_init__(self):
        # Prefer chat turn-end token for MiniCPM-V decoding.
        # Using <|endoftext|> (248044) as EOS can cause immediate empty stop.
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [int(self.eos_token_id)]
        if isinstance(self.eos_token_id, list):
            eos = [int(token_id) for token_id in self.eos_token_id]
            if 248046 in eos:
                self.eos_token_id = [248046]
            elif 248044 in eos and len(eos) == 1:
                self.eos_token_id = []

    @classmethod
    def from_dict(cls, params):
        source_params = params if isinstance(params, dict) else None
        params = dict(params)

        # MiniCPM-V config keeps most LLM fields at the root. Build text_config
        # from root values when explicit text_config is absent.
        text_params = params.pop("text_config", None)
        if not text_params:
            text_fields = {
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "rms_norm_eps",
                "vocab_size",
                "num_key_value_heads",
                "head_dim",
                "rope_theta",
                "max_position_embeddings",
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "full_attention_interval",
                "rope_scaling",
                "rope_parameters",
                "tie_word_embeddings",
                "attention_bias",
                "hidden_act",
            }
            text_params = {k: v for k, v in params.items() if k in text_fields}
        if source_params is not None:
            source_params["text_config"] = dict(text_params)
        text_config = TextConfig.from_dict(text_params)

        vision_params = dict(params.pop("vision_config", {}))
        if vision_params.get("model_type") == "siglip":
            vision_params["model_type"] = "siglip_vision_model"
        if source_params is not None:
            source_params["vision_config"] = dict(vision_params)
        vision_config = VisionConfig.from_dict(vision_params)

        slice_params = params.pop("slice_config", None)
        slice_config = (
            SliceConfig.from_dict(slice_params)
            if isinstance(slice_params, dict)
            else slice_params
        )

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            slice_config=slice_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )
