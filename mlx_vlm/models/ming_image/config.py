"""Configuration for the Ming-Image-0.1-Design text-to-image model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlx_vlm.models.bailing_moe.config import ModelConfig as BailingMoeConfig


def _load(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing Ming-Image config: {path}")
    return json.loads(path.read_text())


def _bailing_config(llm: dict[str, Any]) -> BailingMoeConfig:
    """Map the mllm ``llm_config`` JSON onto the shared BailingMoeV2 config."""
    return BailingMoeConfig(
        model_type="bailing_moe_v2",
        hidden_size=llm["hidden_size"],
        intermediate_size=llm["intermediate_size"],
        max_position_embeddings=llm.get("max_position_embeddings", 32768),
        moe_intermediate_size=llm["moe_intermediate_size"],
        num_experts=llm["num_experts"],
        num_shared_experts=llm["num_shared_experts"],
        norm_topk_prob=llm.get("norm_topk_prob", True),
        num_attention_heads=llm["num_attention_heads"],
        num_experts_per_tok=llm["num_experts_per_tok"],
        num_hidden_layers=llm["num_hidden_layers"],
        num_key_value_heads=llm["num_key_value_heads"],
        rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
        rope_theta=llm.get("rope_theta", 600000.0),
        vocab_size=llm["vocab_size"],
        first_k_dense_replace=llm.get("first_k_dense_replace", 1),
        rope_scaling=None,
        use_qk_norm=True,
        partial_rotary_factor=llm.get("partial_rotary_factor", 0.5),
        moe_router_enable_expert_bias=llm.get("use_expert_bias", True),
        routed_scaling_factor=llm.get(
            "routed_scaling_factor", llm.get("moe_router_topk_scaling_factor", 2.5)
        ),
        score_function="sigmoid",
        n_group=llm.get("n_group", 8),
        topk_group=llm.get("topk_group", 4),
    )


@dataclass(frozen=True, slots=True)
class MingImageDiTConfig:
    dim: int = 3840
    n_heads: int = 30
    n_kv_heads: int = 30
    n_layers: int = 30
    n_refiner_layers: int = 2
    intermediate_size: int = 10240
    cap_feat_dim: int = 2560
    in_channels: int = 16
    patch_size: int = 2
    f_patch_size: int = 1
    axes_dims: tuple[int, ...] = (32, 48, 48)
    rope_theta: float = 256.0
    norm_eps: float = 1e-5
    t_scale: float = 1000.0
    adaln_embed_dim: int = 256

    @classmethod
    def from_dict(cls, c: dict[str, Any]) -> MingImageDiTConfig:
        d = cls()
        dim = int(c.get("dim", d.dim))
        return cls(
            dim=dim,
            n_heads=int(c.get("n_heads", d.n_heads)),
            n_kv_heads=int(c.get("n_kv_heads", d.n_kv_heads)),
            n_layers=int(c.get("n_layers", d.n_layers)),
            n_refiner_layers=int(c.get("n_refiner_layers", d.n_refiner_layers)),
            intermediate_size=int(c.get("intermediate_size", dim * 8 // 3)),
            cap_feat_dim=int(c.get("cap_feat_dim", d.cap_feat_dim)),
            in_channels=int(c.get("in_channels", d.in_channels)),
            patch_size=int(c.get("all_patch_size", (d.patch_size,))[0]),
            f_patch_size=int(c.get("all_f_patch_size", (d.f_patch_size,))[0]),
            axes_dims=tuple(c.get("axes_dims", d.axes_dims)),
            rope_theta=float(c.get("rope_theta", d.rope_theta)),
            norm_eps=float(c.get("norm_eps", d.norm_eps)),
            t_scale=float(c.get("t_scale", d.t_scale)),
        )


@dataclass(frozen=True, slots=True)
class MingImageConnectorConfig:
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, c: dict[str, Any]) -> MingImageConnectorConfig:
        d = cls()
        heads = int(c.get("num_attention_heads", d.num_attention_heads))
        hidden = int(c.get("hidden_size", d.hidden_size))
        return cls(
            hidden_size=hidden,
            num_hidden_layers=int(c.get("num_hidden_layers", d.num_hidden_layers)),
            num_attention_heads=heads,
            num_key_value_heads=int(
                c.get("num_key_value_heads", d.num_key_value_heads)
            ),
            head_dim=int(c.get("head_dim", hidden // heads)),
            intermediate_size=int(c.get("intermediate_size", d.intermediate_size)),
            rope_theta=float(c.get("rope_theta", d.rope_theta)),
            rms_norm_eps=float(c.get("rms_norm_eps", d.rms_norm_eps)),
        )


@dataclass(frozen=True, slots=True)
class MingImageBridgeConfig:
    query_token_count: int = 256
    mllm_hidden: int = 2048
    connector_hidden: int = 1536
    cap_feat_dim: int = 2560
    directvlm_dim: int = 3840
    selected_hidden_states_layers: tuple[int, ...] = (5, 12, 20)

    @property
    def directvlm_in(self) -> int:
        return self.mllm_hidden * len(self.selected_hidden_states_layers)

    @classmethod
    def from_dict(
        cls, c: dict[str, Any], *, mllm_hidden: int, connector_hidden: int
    ) -> MingImageBridgeConfig:
        d = cls()
        scale = int(c.get("img_gen_scales", [16])[0])
        return cls(
            query_token_count=scale * scale,
            mllm_hidden=mllm_hidden,
            connector_hidden=connector_hidden,
            cap_feat_dim=int(c.get("diffusion_c_input_dim", d.cap_feat_dim)),
            directvlm_dim=int(c.get("diffusion_inner_dim", d.directvlm_dim)),
            selected_hidden_states_layers=tuple(
                c.get("selected_hidden_states_layers", d.selected_hidden_states_layers)
            ),
        )


@dataclass(frozen=True, slots=True)
class MingImageVAEConfig:
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    in_channels: int = 4
    out_channels: int = 4
    is_residual: bool = False
    scaling_factor: float = 8.0064
    shift_factor: float = 0.0

    @classmethod
    def from_dict(cls, c: dict[str, Any]) -> MingImageVAEConfig:
        d = cls()
        channels = int(c.get("input_channels", c.get("in_channels", d.in_channels)))
        return cls(
            base_dim=int(c.get("base_dim", d.base_dim)),
            z_dim=int(c.get("z_dim", d.z_dim)),
            dim_mult=tuple(c.get("dim_mult", d.dim_mult)),
            num_res_blocks=int(c.get("num_res_blocks", d.num_res_blocks)),
            temperal_downsample=tuple(
                c.get("temperal_downsample", d.temperal_downsample)
            ),
            in_channels=channels,
            out_channels=int(c.get("out_channels", channels)),
            is_residual=bool(c.get("is_residual", d.is_residual)),
            scaling_factor=float(c.get("scaling_factor", d.scaling_factor)),
            shift_factor=float(c.get("shift_factor", d.shift_factor)),
        )


@dataclass(frozen=True, slots=True)
class MingImageConfig:
    dit: MingImageDiTConfig = field(default_factory=MingImageDiTConfig)
    connector: MingImageConnectorConfig = field(
        default_factory=MingImageConnectorConfig
    )
    bridge: MingImageBridgeConfig = field(default_factory=MingImageBridgeConfig)
    vae: MingImageVAEConfig = field(default_factory=MingImageVAEConfig)
    mllm: BailingMoeConfig | None = None
    image_patch_token: int = 157157
    image_start_token: int = 157158
    image_end_token: int = 157159
    default_steps: int = 12
    default_guidance: float = 1.0
    num_train_timesteps: int = 1000

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> MingImageConfig:
        root = Path(model_path).expanduser()
        llm = _load(root, "mllm/config.json")["llm_config"]
        connector = MingImageConnectorConfig.from_dict(
            _load(root, "connector/config.json")
        )
        scheduler = _load(root, "scheduler/scheduler_config.json")
        return cls(
            dit=MingImageDiTConfig.from_dict(_load(root, "transformer/config.json")),
            connector=connector,
            bridge=MingImageBridgeConfig.from_dict(
                _load(root, "mlp/config.json"),
                mllm_hidden=llm["hidden_size"],
                connector_hidden=connector.hidden_size,
            ),
            vae=MingImageVAEConfig.from_dict(_load(root, "vae/config.json")),
            mllm=_bailing_config(llm),
            image_patch_token=int(llm.get("image_patch_token", 157157)),
            image_start_token=int(llm.get("image_start_token", 157158)),
            image_end_token=int(llm.get("image_end_token", 157159)),
            num_train_timesteps=int(scheduler.get("num_train_timesteps", 1000)),
        )


def detect_ming_image_layout(path: str | Path) -> bool:
    root = Path(path).expanduser()
    transformer = root / "transformer" / "config.json"
    mllm = root / "mllm" / "config.json"
    if not transformer.exists() or not mllm.exists():
        return False
    try:
        dit = json.loads(transformer.read_text())
        backbone = json.loads(mllm.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return dit.get("_class_name") == "DiffusionTransformer" and "BailingMM" in str(
        backbone.get("architectures", [""])[0]
    )


def validate_dimensions(*, width: int, height: int) -> None:
    """Reject sizes outside the model's [256, 2048] multiple-of-16 range."""
    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")


__all__ = [
    "MingImageBridgeConfig",
    "MingImageConfig",
    "MingImageConnectorConfig",
    "MingImageDiTConfig",
    "MingImageVAEConfig",
    "detect_ming_image_layout",
    "validate_dimensions",
]
