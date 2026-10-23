"""Hi3D v1.0 model configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from ..base import BaseModelConfig

LATENT_MEAN = [
    -0.00155994,
    0.03314649,
    -0.00307796,
    0.00630751,
    -0.00835457,
    -0.03860179,
    0.02528263,
    0.03740620,
    -0.06209833,
    -0.00539356,
    -0.02110271,
    0.05126001,
    0.00753371,
    0.02866821,
    0.02772475,
    -0.02131735,
    -0.02799912,
    0.01084617,
    0.03433723,
    -0.00903100,
    -0.00121899,
    -0.01944244,
    0.01641873,
    -0.01063657,
    -0.01990618,
    0.04856062,
    0.01747232,
    -0.02387775,
    0.03216169,
    -0.03732105,
    -0.03730354,
    -0.01948937,
]

LATENT_STD = [
    0.61686146,
    0.80115320,
    0.85489200,
    0.96384716,
    0.77005910,
    0.62859850,
    0.61541015,
    0.74368310,
    0.62762610,
    1.01842280,
    0.77113867,
    0.62627990,
    0.68749195,
    0.62698966,
    0.75491240,
    0.61577400,
    0.63780660,
    0.80406000,
    0.83846325,
    0.69120390,
    0.69227580,
    0.79321617,
    0.63475930,
    0.63848300,
    0.67194235,
    0.73519380,
    0.84781533,
    0.93012910,
    0.62691440,
    0.67906080,
    0.65198770,
    0.77692040,
]


@dataclass
class TransformerConfig(BaseModelConfig):
    """One sparse latent flow transformer (``transformer_512`` or ``_1024``)."""

    resolution: int = 64
    in_channels: int = 32
    out_channels: int = 32
    model_channels: int = 2112
    cond_channels: int = 2048
    num_blocks: int = 24
    num_heads: int = 22
    mlp_ratio: float = 4.0
    patch_size: int = 2
    num_io_res_blocks: int = 2
    io_block_channels: List[int] = field(default_factory=lambda: [256])
    latent_mean: List[float] = field(default_factory=lambda: list(LATENT_MEAN))
    latent_std: List[float] = field(default_factory=lambda: list(LATENT_STD))

    def __post_init__(self):
        if self.model_channels % self.num_heads:
            raise ValueError("model_channels must be divisible by num_heads")
        if (self.model_channels // self.num_heads) % 6:
            raise ValueError("the head dimension must split into three RoPE axes")
        if self.patch_size != 2 ** len(self.io_block_channels):
            raise ValueError("patch_size must equal 2 ** len(io_block_channels)")
        if not len(self.latent_mean) == len(self.latent_std) == self.in_channels:
            raise ValueError("latent statistics must have in_channels entries")


@dataclass
class DecoderConfig(BaseModelConfig):
    """Sparse Point Transformer V3 mesh decoder (the released ``vae``)."""

    z_channels: int = 32
    out_channels: int = 32
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: List[int] = field(default_factory=lambda: [128, 256, 512, 512])
    mid_depth: int = 2
    mid_num_head: int = 32
    patch_size: int = 48
    mlp_ratio: float = 4.0
    order: List[str] = field(default_factory=lambda: ["z", "z-trans"])

    def __post_init__(self):
        if len(self.depths) != len(self.channels):
            raise ValueError("depths and channels must describe the same stages")


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "dinov2"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "none"


def _dino_config():
    return {
        "model_type": "dinov2",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "layer_norm_eps": 1e-6,
        "image_size": 518,
        "patch_size": 14,
        "layerscale_value": 1.0,
    }


def _clip_config():
    return {
        "model_type": "clip_vision_model",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "projection_dim": 768,
        "layer_norm_eps": 1e-5,
    }


@dataclass
class Hi3DConfig(BaseModelConfig):
    model_type: str = "hi3d"

    transformer_512: TransformerConfig = field(default_factory=TransformerConfig)
    transformer_1024: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(resolution=128)
    )
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Image conditioning: L2-normalized DINOv2-L tokens at 756 px, each joined
    # with the L2-normalized CLIP ViT-L/14 pooled embedding at 224 px.
    dino_config: dict = field(default_factory=_dino_config)
    clip_config: dict = field(default_factory=_clip_config)
    dino_image_size: int = 756
    clip_image_size: int = 224

    # Flow sampling: Heun steps per pass and the released schedule constants.
    num_inference_steps: int = 25
    sigma_min: float = 0.002
    time_rescale_factor: float = 2.0

    # Refinement ladder: structure grid, output mesh resolution, occupied-voxel
    # budget and final face target of each pass.
    structure_resolutions: List[int] = field(default_factory=lambda: [64, 128, 192])
    output_resolutions: List[int] = field(default_factory=lambda: [512, 1024, 1536])
    pass_budgets: List[int] = field(default_factory=lambda: [80000, 90000, 110000])
    target_face_counts: List[int] = field(
        default_factory=lambda: [1000000, 1000000, 2000000]
    )

    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)

    def __post_init__(self):
        for name in ("transformer_512", "transformer_1024"):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, TransformerConfig.from_dict(value))
        if isinstance(self.decoder, dict):
            self.decoder = DecoderConfig.from_dict(self.decoder)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        ladders = (
            self.structure_resolutions,
            self.output_resolutions,
            self.pass_budgets,
            self.target_face_counts,
        )
        if len({len(ladder) for ladder in ladders}) != 1:
            raise ValueError("refinement ladder lists must have equal lengths")

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        with open(Path(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: Path) -> "Hi3DConfig":
        with open(Path(path)) as f:
            return cls.from_dict(json.load(f))


ModelConfig = Hi3DConfig
