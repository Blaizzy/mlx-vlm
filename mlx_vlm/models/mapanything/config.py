"""MapAnything configuration.

Reads the ``config.json`` of the official checkpoints (``facebook/map-anything``,
``facebook/map-anything-apache`` and their ``-v1`` releases) unchanged: the
nested UniCeption ``*_config`` dicts become the typed sections below. Only the
architecture of those releases is supported (DINOv2 encoder, alternating-
attention transformer, DPT + pose heads with the ray-directions + depth +
pose scene representation); other UniCeption options raise ``ValueError``.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig
from ..dinov2.config import DINOV2_PRESETS

_DINOV2_SIZES = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}

SCENE_REPRESENTATIONS = (
    "raydirs+depth+pose",
    "raydirs+depth+pose+confidence",
    "raydirs+depth+pose+mask",
    "raydirs+depth+pose+confidence+mask",
)


def _require(value, expected, name):
    if value != expected:
        raise ValueError(
            f"MapAnything: unsupported {name} {value!r}; expected {expected!r}"
        )


@dataclass
class EncoderConfig(BaseModelConfig):
    encoder_str: str = "dinov2"
    size: str = "large"
    patch_size: int = 14
    with_registers: bool = False
    keep_first_n_layers: Optional[int] = None
    norm_returned_features: bool = True
    data_norm_type: str = "dinov2"

    def __post_init__(self):
        _require(self.encoder_str, "dinov2", "encoder")
        if self.size not in _DINOV2_SIZES:
            raise ValueError(f"MapAnything: unknown DINOv2 size {self.size!r}")
        preset = DINOV2_PRESETS[_DINOV2_SIZES[self.size]]
        self.embed_dim = preset["embed_dim"]
        self.num_heads = preset["num_heads"]
        self.ffn = preset["ffn"]
        self.depth = self.keep_first_n_layers or preset["depth"]
        self.mlp_ratio = 4.0
        self.layer_norm_eps = 1e-6
        self.img_size = 518
        self.num_register_tokens = 4 if self.with_registers else 0
        # torch.hub DINOv2 position-embedding interpolation settings.
        self.interpolate_offset = 0.0 if self.with_registers else 0.1
        self.interpolate_antialias = self.with_registers


@dataclass
class InfoSharingConfig(BaseModelConfig):
    model_type: str = "alternating_attention"
    model_return_type: str = "intermediate_features"
    custom_positional_encoding: Optional[str] = None
    module_args: dict = field(
        default_factory=lambda: dict(
            depth=16, dim=1536, num_heads=24, indices=[7, 11], init_values=1e-5
        )
    )

    def __post_init__(self):
        _require(self.model_type, "alternating_attention", "info sharing model")
        _require(self.model_return_type, "intermediate_features", "info sharing return")
        _require(self.custom_positional_encoding, None, "positional encoding")
        args = self.module_args
        for name, default in (
            ("qk_norm", False),
            ("use_pe_for_non_reference_views", False),
        ):
            _require(args.get(name, default), default, name)
        self.depth = args.get("depth", 12)
        self.dim = args.get("dim", 768)
        self.num_heads = args.get("num_heads", 12)
        self.mlp_ratio = args.get("mlp_ratio", 4.0)
        self.qkv_bias = args.get("qkv_bias", True)
        self.layer_scale = bool(args.get("init_values"))
        self.indices = list(args.get("indices", []))
        self.norm_intermediate = args.get("norm_intermediate", True)
        self.distinguish_ref_and_non_ref_views = args.get(
            "distinguish_ref_and_non_ref_views", True
        )
        if len(self.indices) != 2:
            raise ValueError("MapAnything: the DPT head expects 2 intermediate layers")


@dataclass
class PredHeadConfig(BaseModelConfig):
    type: str = "dpt+pose"
    adaptor_type: str = "raydirs+depth+pose+confidence+mask"
    feature_head: dict = field(default_factory=dict)
    regressor_head: dict = field(default_factory=dict)
    pose_head: dict = field(default_factory=dict)
    scale_head: dict = field(default_factory=dict)
    dpt_adaptor: dict = field(default_factory=dict)
    pose_adaptor: dict = field(default_factory=dict)
    scale_adaptor: dict = field(default_factory=dict)

    def __post_init__(self):
        _require(self.type, "dpt+pose", "prediction head")
        if self.adaptor_type not in SCENE_REPRESENTATIONS:
            raise ValueError(
                f"MapAnything: unsupported adaptor {self.adaptor_type!r}; "
                f"expected one of {SCENE_REPRESENTATIONS}"
            )
        self.has_confidence = "confidence" in self.adaptor_type
        self.has_mask = "mask" in self.adaptor_type
        self.feature_dim = self.feature_head.get("feature_dim", 256)
        self.layer_dims = self.feature_head.get("layer_dims", [96, 192, 384, 768])
        self.dense_channels = 4 + self.has_confidence + self.has_mask
        output_dim = self.regressor_head.get("output_dim", self.dense_channels)
        _require(output_dim, self.dense_channels, "dense output channels")
        self.num_resconv_block = self.pose_head.get("num_resconv_block", 2)
        self.rot_representation_dim = self.pose_head.get("rot_representation_dim", 4)
        self.scale_hidden_dim = self.scale_head.get("hidden_dim", 196)
        self.scale_num_mlp_layers = self.scale_head.get("num_mlp_layers", 2)


@dataclass
class GeometricInputConfig(BaseModelConfig):
    ray_dirs_encoder_config: dict = field(default_factory=lambda: {"apply_pe": False})
    depth_encoder_config: dict = field(default_factory=lambda: {"apply_pe": False})
    scale_encoder_config: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("ray_dirs_encoder_config", "depth_encoder_config"):
            _require(
                getattr(self, name).get("apply_pe", True), False, f"{name}.apply_pe"
            )
        self.dense_intermediate_dims = self.ray_dirs_encoder_config.get(
            "intermediate_dims", [588, 768, 1024]
        )
        self.global_intermediate_dims = self.scale_encoder_config.get(
            "intermediate_dims", [128, 256, 512]
        )


_SECTIONS = {
    "encoder_config": EncoderConfig,
    "info_sharing_config": InfoSharingConfig,
    "pred_head_config": PredHeadConfig,
    "geometric_input_config": GeometricInputConfig,
}


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "mapanything"
    encoder_config: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            size="giant", keep_first_n_layers=24, norm_returned_features=False
        )
    )
    info_sharing_config: InfoSharingConfig = field(default_factory=InfoSharingConfig)
    pred_head_config: PredHeadConfig = field(default_factory=PredHeadConfig)
    geometric_input_config: GeometricInputConfig = field(
        default_factory=GeometricInputConfig
    )
    use_register_tokens_from_encoder: bool = True
    info_sharing_mlp_layer_str: str = "swiglufused"
    # Load dtype of the geometric encoders and heads (None keeps the checkpoint's).
    head_dtype: Optional[str] = "float32"
    # Views per dense-head pass (0 for all at once); bounds full-resolution conv memory.
    dense_head_chunk_size: int = 4

    def __post_init__(self):
        if self.info_sharing_mlp_layer_str not in ("mlp", "swiglufused"):
            raise ValueError(
                f"MapAnything: unknown MLP {self.info_sharing_mlp_layer_str!r}"
            )

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        for key, section in _SECTIONS.items():
            if isinstance(params.get(key), dict):
                params[key] = section.from_dict(params[key])
        return super().from_dict(params)

    @property
    def dpt_input_dims(self) -> List[int]:
        dim = self.info_sharing_config.dim
        return [self.encoder_config.embed_dim, dim, dim, dim]

    @property
    def pose_head_dim(self) -> int:
        return 4 * self.encoder_config.patch_size**2
