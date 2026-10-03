"""DINOv2 architecture presets and standalone model configuration."""

from dataclasses import dataclass
from typing import List, Union

from ..base import BaseModelConfig

DINOV2_PRESETS = {
    "vits14": dict(embed_dim=384, depth=12, num_heads=6, ffn="mlp"),
    "vitb14": dict(embed_dim=768, depth=12, num_heads=12, ffn="mlp"),
    "vitl14": dict(embed_dim=1024, depth=24, num_heads=16, ffn="mlp"),
    "vitg14": dict(embed_dim=1536, depth=40, num_heads=24, ffn="swiglu"),
}


@dataclass
class ModelConfig(BaseModelConfig):
    """Standalone DINOv2 model configuration.

    Field names follow the Hugging Face ``dinov2`` and
    ``dinov2_with_registers`` configs so Hub checkpoints load unchanged.
    """

    model_type: str = "dinov2"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    image_size: Union[int, List[int]] = 224
    patch_size: int = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 0
    # Pos-embed interpolation follows the HF reference (size-based) by default;
    # set interpolate_offset=0.1 for the original repo's scale-factor behavior.
    interpolate_offset: float = 0.0
    interpolate_antialias: bool = False

    def __post_init__(self):
        if isinstance(self.image_size, (list, tuple)):
            self.image_size = self.image_size[0]

    # Aliases used by the shared backbone (DINOv2 / DINOv2Encoder).
    @property
    def embed_dim(self) -> int:
        return self.hidden_size

    @property
    def depth(self) -> int:
        return self.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def img_size(self) -> int:
        return self.image_size

    @property
    def ffn(self) -> str:
        return "swiglu" if self.use_swiglu_ffn else "mlp"
