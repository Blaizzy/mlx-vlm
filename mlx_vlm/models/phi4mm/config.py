from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "phi4mm"
    max_position_embeddings: int = 131072


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip2_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    image_size: int = 448
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class AudioConfig:
    """Configuration for the Cascades Conformer audio encoder."""

    attention_dim: int = 1024
    attention_heads: int = 16
    num_blocks: int = 24
    linear_units: int = 1536
    input_size: int = 80
    time_reduction: int = 8
    kernel_size: int = 3
    dropout_rate: float = 0.0
    activation: str = "swish"
    conv_activation: str = "swish"
    conv_glu_type: str = "swish"
    bias_in_glu: bool = True
    ext_pw_out_channel: int = 1024
    ext_pw_kernel_size: int = 1
    depthwise_seperable_out_channel: int = 1024
    depthwise_multiplier: int = 1
    causal: bool = True
    batch_norm: bool = False
    cnn_layer_norm: bool = True
    t5_bias_max_distance: int = 500
    conv_channels: int = 1024
    chunk_size: int = -1
    left_chunk: int = 18


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "phi4mm"
    vocab_size: int = 200064
    hidden_size: int = 3072
    num_hidden_layers: int = 32
    intermediate_size: int = 8192
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    partial_rotary_factor: float = 0.75
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    mm_hidden_size: int = 1152
    mm_projector_type: str = "mlp2x_gelu"
    image_token_index: int = -200
    audio_token_index: int = 200011
    pad_token_id: int = 199999
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = True
    vision_lora: Optional[Dict] = None
    speech_lora: Optional[Dict] = None
    audio_processor: Optional[Dict] = None

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        # Parse audio config from audio_processor dict
        if self.audio_processor and isinstance(self.audio_processor, dict):
            self._audio_config = AudioConfig(
                **{
                    k: v
                    for k, v in self.audio_processor.get("config", {}).items()
                    if k in AudioConfig.__dataclass_fields__
                }
            )
        else:
            self._audio_config = AudioConfig()
