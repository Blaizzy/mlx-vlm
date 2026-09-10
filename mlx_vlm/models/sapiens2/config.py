"""Sapiens2 configuration (backbone + per-task head)."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..base import BaseModelConfig

# Arch zoo mirrored from sapiens/backbones/sapiens2.py:arch_zoo
ARCH_ZOO = {
    "sapiens2_0.1b": {"embed_dims": 768, "num_layers": 12, "num_heads": 12},
    "sapiens2_0.4b": {"embed_dims": 1024, "num_layers": 24, "num_heads": 16},
    "sapiens2_0.8b": {"embed_dims": 1280, "num_layers": 32, "num_heads": 16},
    "sapiens2_1b": {"embed_dims": 1536, "num_layers": 40, "num_heads": 24},
    "sapiens2_5b": {"embed_dims": 2432, "num_layers": 56, "num_heads": 32},
}

# Per-task per-size head defaults, pulled from sapiens/{pose,dense}/configs/.../sapiens2_<size>_<task>-1024x768.py
# Missing size entries fall back to the 0.4b shape; override via explicit HeadConfig when needed.
HEAD_DEFAULTS = {
    "pose": {
        "0.4b": {
            "deconv_out_channels": (1024, 768),
            "deconv_kernel_sizes": (4, 4),
            "conv_out_channels": (512, 512, 256),
            "conv_kernel_sizes": (1, 1, 1),
            "num_keypoints": 308,
        },
    },
    "seg": {
        "0.4b": {
            "deconv_out_channels": (512, 256, 128, 64),
            "deconv_kernel_sizes": (4, 4, 4, 4),
            "conv_out_channels": (64, 64),
            "conv_kernel_sizes": (1, 1),
            "num_classes": 29,
        },
    },
    "normal": {
        "0.4b": {
            "upsample_channels": (768, 512, 256, 128),
            "conv_out_channels": (64, 32, 16),
            "conv_kernel_sizes": (3, 3, 3),
        },
    },
    "pointmap": {
        "0.4b": {
            "upsample_channels": (1536, 768, 512, 256),
            "conv_out_channels": (64, 32, 16),
            "conv_kernel_sizes": (3, 3, 3),
            "scale_conv_out_channels": (1536, 512, 128),
            "scale_conv_kernel_sizes": (1, 1, 1),
            "scale_final_layer": (6144, 512, 128, 1),
        },
    },
}

TASKS = ("pose", "seg", "normal", "pointmap")
SIZES = ("0.1b", "0.4b", "0.8b", "1b", "5b")


@dataclass
class BackboneConfig(BaseModelConfig):
    model_type: str = "sapiens2_backbone"
    arch: str = "sapiens2_0.4b"
    embed_dims: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    feedforward_channels: int = 4096
    image_size: Tuple[int, int] = (1024, 768)
    patch_size: int = 16
    n_storage_tokens: int = 8
    rope_base: float = 100.0
    rope_normalize_coords: str = "separate"
    mhsa_early: int = 8
    mhsa_late: int = 8
    layer_scale_init_value: float = 1e-4
    final_norm: bool = True


@dataclass
class HeadConfig(BaseModelConfig):
    """Union type across the four task heads; task selects which fields are used."""

    model_type: str = "sapiens2_head"
    task: str = "seg"  # one of TASKS
    in_channels: int = 1024

    # pose + seg
    deconv_out_channels: Optional[List[int]] = None
    deconv_kernel_sizes: Optional[List[int]] = None
    conv_out_channels: Optional[List[int]] = None
    conv_kernel_sizes: Optional[List[int]] = None

    # pose
    num_keypoints: int = 308

    # seg
    num_classes: int = 29

    # normal + pointmap
    upsample_channels: Optional[List[int]] = None

    # pointmap
    scale_conv_out_channels: Optional[List[int]] = None
    scale_conv_kernel_sizes: Optional[List[int]] = None
    scale_final_layer: Optional[List[int]] = None


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "sapiens2"
    task: str = "seg"  # pose | seg | normal | pointmap
    size: str = "0.4b"  # 0.1b | 0.4b | 0.8b | 1b | 5b
    backbone_config: Optional[dict] = None
    head_config: Optional[dict] = None

    def __post_init__(self):
        assert self.task in TASKS, f"task must be one of {TASKS}, got {self.task}"
        assert self.size in SIZES, f"size must be one of {SIZES}, got {self.size}"

        arch = f"sapiens2_{self.size}"
        arch_params = ARCH_ZOO[arch]

        if self.backbone_config is None:
            self.backbone_config = BackboneConfig(
                arch=arch,
                embed_dims=arch_params["embed_dims"],
                num_layers=arch_params["num_layers"],
                num_heads=arch_params["num_heads"],
                feedforward_channels=arch_params["embed_dims"] * 4,
            )
        elif isinstance(self.backbone_config, dict):
            self.backbone_config = BackboneConfig.from_dict(self.backbone_config)

        if self.head_config is None:
            head_defaults = HEAD_DEFAULTS[self.task].get(
                self.size, HEAD_DEFAULTS[self.task]["0.4b"]
            )
            self.head_config = HeadConfig(
                task=self.task,
                in_channels=arch_params["embed_dims"],
                **head_defaults,
            )
        elif isinstance(self.head_config, dict):
            self.head_config = HeadConfig.from_dict(self.head_config)

        self.text_config = None
        self.vision_config = None
