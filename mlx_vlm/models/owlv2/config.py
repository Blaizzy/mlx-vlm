"""OWLv2 configuration."""

from dataclasses import dataclass, field

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "owlv2_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int = 960
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "owlv2_text"
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 16
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "owlv2"
    projection_dim: int = 512
    num_classes: int = 0  # open-vocabulary, set by text queries
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
