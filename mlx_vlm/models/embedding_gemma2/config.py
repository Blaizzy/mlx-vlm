from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig
from ..gemma4.config import AudioConfig, VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "embedding_gemma2_text"
    vocab_size: int = 262144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_size_per_layer_input: int = 512
    embedding_dim: int = 768
    rms_norm_eps: float = 1e-6
    sliding_window: int = 512
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"
    pad_token_id: int = 0
    layer_types: Optional[List[str]] = None
    per_layer_config: Dict = field(default_factory=dict)
    rope_parameters: Optional[Dict] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per text layer")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_theta": 1_000_000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            }


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "embedding_gemma2"
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None
    audio_config: Optional[AudioConfig] = None
    image_token_id: int = 258880
    audio_token_id: int = 258881
    video_token_id: int = 258884
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    boa_token_id: int = 256000
    eoa_token_index: int = 258883
    vision_soft_tokens_per_image: int = 280

    def __post_init__(self):
        for name, cls in (
            ("text_config", TextConfig),
            ("vision_config", VisionConfig),
            ("audio_config", AudioConfig),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls.from_dict(value))
