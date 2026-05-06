"""SAM 3D Body model configuration."""

import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class BaseModelConfig:
    """Base config with from_dict() for mlx-vlm compatibility."""

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class VisionConfig(BaseModelConfig):
    """Vision encoder config (DINOv3 ViT-H+)."""

    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 20
    head_dim: int = 64
    patch_size: int = 16
    image_size: Tuple[int, int] = (512, 384)
    ffn_ratio: float = 4.0
    num_storage_tokens: int = 4
    rope_periods: int = 16
    drop_path_rate: float = 0.1
    model_type: str = "dinov3"


@dataclass
class TextConfig(BaseModelConfig):
    """Stub text config — SAM 3D Body has no text encoder."""

    model_type: str = "none"


@dataclass
class SAM3DConfig(BaseModelConfig):
    # Backbone (DINOv3 ViT-H+)
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 20
    head_dim: int = 64
    patch_size: int = 16
    image_size: Tuple[int, int] = (512, 384)
    ffn_ratio: float = 4.0
    num_storage_tokens: int = 4
    rope_periods: int = 16
    drop_path_rate: float = 0.1

    # Decoder
    decoder_dim: int = 1024
    decoder_depth: int = 6
    decoder_heads: int = 8
    decoder_head_dim: int = 64
    decoder_mlp_dim: int = 1024

    # MHR character model
    num_joints: int = 127
    num_vertices: int = 18439
    num_faces: int = 36874
    num_shape_comps: int = 45
    num_face_comps: int = 72

    # Pose head output dim: 519 = body_cont_dim + ...
    pose_output_dim: int = 519
    camera_output_dim: int = 3

    # Prompt encoder
    num_point_embeddings: int = 70
    prompt_embed_dim: int = 1280

    # Image normalization (ImageNet)
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Modes
    enable_body: bool = True
    enable_hand: bool = True
    model_type: str = "sam3d_body"

    # Sub-configs (populated by from_dict for mlx-vlm compat)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        vision_params = params.pop("vision_config", {})
        text_params = params.pop("text_config", {})
        filtered = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        return cls(
            vision_config=VisionConfig.from_dict(vision_params),
            text_config=TextConfig.from_dict(text_params),
            **filtered,
        )

    def save(self, path: Path):
        path = Path(path)
        d = asdict(self)
        # Convert tuples to lists for JSON
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SAM3DConfig":
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        # Convert lists back to tuples for tuple fields
        tuple_fields = {"image_size", "image_mean", "image_std"}
        for k in tuple_fields:
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)


# mlx-vlm convention alias
ModelConfig = SAM3DConfig
