"""Sapiens2 configuration (inference only).

Defaults reproduce the Sapiens2-1B backbone; every field has a default so
partial configs still load. Training-only keys of the HF config (dropout,
drop path, initializer range, mask token, position jitter) are ignored.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig

# Architecture table from the original repo (used for tokenizer defaults and
# convenience). Keyed by arch name: (hidden_size, layers, heads, tok_layers).
SAPIENS2_ARCH = {
    "sapiens2_0.1b": (768, 12, 12, 2),
    "sapiens2_0.4b": (1024, 24, 16, 2),
    "sapiens2_0.8b": (1280, 32, 16, 3),
    "sapiens2_1b": (1536, 40, 24, 4),
    "sapiens2_5b": (2432, 56, 32, 6),
}

# Task name for each ``architectures`` entry used by the official repos.
ARCHITECTURE_TASKS = {
    "Sapiens2Model": "backbone",
    "Sapiens2Backbone": "backbone",
    "Sapiens2ForSemanticSegmentation": "seg",
    "Sapiens2ForPoseEstimation": "pose",
    "Sapiens2ForNormalEstimation": "normal",
    "Sapiens2ForPointmapEstimation": "pointmap",
    "Sapiens2ForImageMatting": "matting",
}


@dataclass
class HeadConfig(BaseModelConfig):
    """Decode-head parameters (``head_config`` in the HF config)."""

    model_type: str = "sapiens2_head"
    # Upsampling: ConvTranspose2d stacks (seg/pose) or Conv2d +
    # PixelShuffle stacks, per use_pixel_shuffle.
    upsample_out_channels: Optional[List[int]] = None
    upsample_kernel_sizes: Optional[List[int]] = None
    use_pixel_shuffle: Optional[bool] = None
    # Optional extra Conv2d stacks after upsampling.
    conv_out_channels: Optional[List[int]] = None
    conv_kernel_sizes: Optional[List[int]] = None
    # Pointmap-only scale regression branch.
    scale_conv_out_channels: Optional[List[int]] = None
    scale_conv_kernel_sizes: Optional[List[int]] = None
    scale_final_input_size: Optional[int] = None
    scale_final_hidden_sizes: Optional[List[int]] = None


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "sapiens2"
    architectures: List[str] = field(default_factory=lambda: ["Sapiens2Model"])

    # Backbone
    hidden_size: int = 1536
    num_hidden_layers: int = 40
    num_attention_heads: int = 24
    # Per-layer KV heads; when None the first/last ``num_*_full_attention_layers``
    # use full MHSA and the middle layers use half the query heads (GQA).
    num_key_value_heads_per_layer: Optional[List[int]] = None
    num_first_full_attention_layers: int = 8
    num_last_full_attention_layers: int = 8
    intermediate_size: int = 6144
    image_size: List[int] = field(default_factory=lambda: [1024, 768])  # (H, W)
    patch_size: int = 16
    num_channels: int = 3
    num_register_tokens: int = 8
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    query_bias: bool = True
    key_bias: bool = True
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    normalize_backbone_outputs: bool = True

    # 4K variant: window-attention tokenizer before the main blocks.
    use_tokenizer: bool = False
    tokenizer_window_size: int = 4
    num_tokenizer_layers: Optional[int] = None  # defaults to the arch table

    # Head
    head_config: Optional[HeadConfig] = None
    num_labels: Optional[int] = None
    id2label: Optional[dict] = None
    flip_pairs: Optional[List[List[int]]] = None  # pose flip-test pairing

    def __post_init__(self):
        if isinstance(self.head_config, dict):
            self.head_config = HeadConfig.from_dict(self.head_config)
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        if self.num_labels is None and self.id2label:
            self.num_labels = len(self.id2label)
        if self.use_tokenizer and self.num_tokenizer_layers is None:
            key = (self.hidden_size, self.num_hidden_layers)
            for _, (d, l, _, t) in SAPIENS2_ARCH.items():
                if (d, l) == key:
                    self.num_tokenizer_layers = t
                    break
            if self.num_tokenizer_layers is None:
                self.num_tokenizer_layers = 4

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        params = dict(params)
        if isinstance(params.get("head_config"), dict):
            params["head_config"] = HeadConfig.from_dict(params["head_config"])
        return super().from_dict(params)

    @property
    def task(self) -> str:
        for arch in self.architectures or ["Sapiens2Model"]:
            if arch in ARCHITECTURE_TASKS:
                return ARCHITECTURE_TASKS[arch]
        return "backbone"

    @property
    def kv_heads_per_layer(self) -> List[int]:
        if self.num_key_value_heads_per_layer is not None:
            return list(self.num_key_value_heads_per_layer)
        heads = []
        for i in range(self.num_hidden_layers):
            if (
                i < self.num_first_full_attention_layers
                or i >= self.num_hidden_layers - self.num_last_full_attention_layers
            ):
                heads.append(self.num_attention_heads)
            else:
                heads.append(self.num_attention_heads // 2)
        return heads

    @property
    def grid_size(self):
        """Patch grid (h, w) at the configured image size."""
        scale = self.tokenizer_window_size if self.use_tokenizer else 1
        return (
            self.image_size[0] // self.patch_size // scale,
            self.image_size[1] // self.patch_size // scale,
        )
