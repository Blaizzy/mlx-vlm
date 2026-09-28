"""SAM 3D Objects model configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ..base import BaseModelConfig

if TYPE_CHECKING:
    from ..moge3.config import ModelConfig as DepthConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "dinov2"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "none"


def _depth_config(depth):
    if isinstance(depth, dict) and depth.get("model_type", "moge3") == "moge3":
        from ..moge3.config import ModelConfig as DepthConfig

        return DepthConfig.from_dict(depth)

    if isinstance(depth, (bool, dict)):
        if depth:
            raise ValueError(
                "depth_model must be a MoGe-3 configuration; bundles converted "
                "with the retired MoGe-v1 depth model need the updated Hub "
                "bundle or convert.py --moge-checkpoint"
            )
        return None

    return depth


@dataclass
class SAM3DObjectsConfig(BaseModelConfig):
    model_type: str = "sam3d_objects"
    dtype: str = "bfloat16"

    # Flow transformers (``ss`` structure/pose stage, ``slat`` feature stage)
    hidden_size: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    cond_channels: int = 1024
    latent_channels: int = 8
    latent_resolution: int = 16
    resolution: int = 64
    io_channels: int = 128

    # Decoders (Gaussian and mesh window transformers, dense structure CNN)
    decoder_channels: int = 768
    decoder_heads: int = 12
    decoder_blocks: int = 12
    window_size: int = 8
    structure_channels: List[int] = field(default_factory=lambda: [512, 128, 32])
    structure_res_blocks: int = 2

    # Condition encoders (DINOv2 ViT-L/14 backbone, point-map patch embedding)
    dino_hidden_size: int = 1024
    dino_heads: int = 16
    dino_layers: int = 24
    dino_image_size: int = 518
    image_size: int = 518
    point_size: int = 256
    point_patch_size: int = 8
    point_channels: int = 512
    point_heads: int = 16

    # Depth estimation (optional MoGe-3, added by convert.py --moge-checkpoint)
    depth_model: Optional["DepthConfig"] = None
    depth_num_tokens: Optional[int] = None

    # Sampling (steps, classifier-free guidance, timestep rescaling)
    ss_steps: int = 25
    slat_steps: int = 25
    ss_guidance: float = 7.0
    slat_guidance: float = 1.0
    ss_rescale_t: float = 3.0
    slat_rescale_t: float = 1.0
    downsample_ss_dist: int = 1

    # Sparse latent normalization (per-channel statistics of the released model)
    slat_mean: List[float] = field(
        default_factory=lambda: [
            0.12211431,
            0.37204156,
            -1.26521907,
            -2.05276058,
            -3.10432536,
            -0.11294304,
            -0.85146744,
            0.45506954,
        ]
    )
    slat_std: List[float] = field(
        default_factory=lambda: [
            2.37326008,
            2.13174402,
            2.2413953,
            2.30589401,
            2.1191894,
            1.8969511,
            2.41684989,
            2.08374642,
        ]
    )

    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)

    def __post_init__(self):
        self.depth_model = _depth_config(self.depth_model)
        if not isinstance(self.vision_config, VisionConfig):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if not isinstance(self.text_config, TextConfig):
            self.text_config = TextConfig.from_dict(self.text_config)

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        with open(Path(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: Path) -> "SAM3DObjectsConfig":
        with open(Path(path)) as f:
            return cls.from_dict(json.load(f))


ModelConfig = SAM3DObjectsConfig
