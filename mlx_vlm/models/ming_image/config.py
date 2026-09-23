"""Configuration for the Ming-Image-0.1-Design text-to-image model.

Ming-Image is a unified understanding+generation model; only the text-to-image
path is ported here. Four weight groups drive it:

* ``mllm`` - a BailingMoeV2 mixture-of-experts LLM that encodes the prompt.
* ``connector`` - a Qwen2 encoder that turns 256 learnable query tokens into
  caption features.
* ``mlp`` - the bridge holding the query tokens and the input/output/direct-VLM
  projections.
* ``transformer`` - a Lumina/NextDiT diffusion transformer.
* ``vae`` - an ``AutoencoderKLQwenImage`` (4-channel RGBA).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing Ming-Image config: {path}")
    return json.loads(path.read_text())


@dataclass(frozen=True, slots=True)
class MingImageDiTConfig:
    dim: int = 3840
    n_heads: int = 30
    n_kv_heads: int = 30
    n_layers: int = 30
    n_refiner_layers: int = 2
    intermediate_size: int = 10240
    cap_feat_dim: int = 2560
    directvlm_dim: int = 3840
    in_channels: int = 16
    patch_size: int = 2
    f_patch_size: int = 1
    axes_dims: tuple[int, ...] = (32, 48, 48)
    rope_theta: float = 256.0
    norm_eps: float = 1e-5
    t_scale: float = 1000.0
    adaln_embed_dim: int = 256

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MingImageDiTConfig:
        defaults = cls()
        dim = int(config.get("dim", defaults.dim))
        return cls(
            dim=dim,
            n_heads=int(config.get("n_heads", defaults.n_heads)),
            n_kv_heads=int(config.get("n_kv_heads", defaults.n_kv_heads)),
            n_layers=int(config.get("n_layers", defaults.n_layers)),
            n_refiner_layers=int(
                config.get("n_refiner_layers", defaults.n_refiner_layers)
            ),
            intermediate_size=int(config.get("intermediate_size", dim * 8 // 3)),
            cap_feat_dim=int(config.get("cap_feat_dim", defaults.cap_feat_dim)),
            directvlm_dim=int(config.get("dim", defaults.directvlm_dim)),
            in_channels=int(config.get("in_channels", defaults.in_channels)),
            patch_size=int(config.get("all_patch_size", (defaults.patch_size,))[0]),
            f_patch_size=int(
                config.get("all_f_patch_size", (defaults.f_patch_size,))[0]
            ),
            axes_dims=tuple(config.get("axes_dims", defaults.axes_dims)),
            rope_theta=float(config.get("rope_theta", defaults.rope_theta)),
            norm_eps=float(config.get("norm_eps", defaults.norm_eps)),
            t_scale=float(config.get("t_scale", defaults.t_scale)),
        )


@dataclass(frozen=True, slots=True)
class MingImageMLLMConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    use_expert_bias: bool = True
    use_qk_norm: bool = True
    partial_rotary_factor: float = 0.5
    rope_theta: float = 600000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    vocab_size: int = 157184
    image_patch_token: int = 157157
    image_start_token: int = 157158
    image_end_token: int = 157159

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MingImageMLLMConfig:
        llm = config.get("llm_config", config)
        defaults = cls()
        return cls(
            hidden_size=int(llm.get("hidden_size", defaults.hidden_size)),
            num_hidden_layers=int(
                llm.get("num_hidden_layers", defaults.num_hidden_layers)
            ),
            num_attention_heads=int(
                llm.get("num_attention_heads", defaults.num_attention_heads)
            ),
            num_key_value_heads=int(
                llm.get("num_key_value_heads", defaults.num_key_value_heads)
            ),
            head_dim=int(llm.get("head_dim", defaults.head_dim)),
            intermediate_size=int(
                llm.get("intermediate_size", defaults.intermediate_size)
            ),
            moe_intermediate_size=int(
                llm.get("moe_intermediate_size", defaults.moe_intermediate_size)
            ),
            num_experts=int(llm.get("num_experts", defaults.num_experts)),
            num_experts_per_tok=int(
                llm.get("num_experts_per_tok", defaults.num_experts_per_tok)
            ),
            num_shared_experts=int(
                llm.get("num_shared_experts", defaults.num_shared_experts)
            ),
            first_k_dense_replace=int(
                llm.get("first_k_dense_replace", defaults.first_k_dense_replace)
            ),
            n_group=int(llm.get("n_group", defaults.n_group)),
            topk_group=int(llm.get("topk_group", defaults.topk_group)),
            routed_scaling_factor=float(
                llm.get(
                    "routed_scaling_factor",
                    llm.get(
                        "moe_router_topk_scaling_factor",
                        defaults.routed_scaling_factor,
                    ),
                )
            ),
            norm_topk_prob=bool(llm.get("norm_topk_prob", defaults.norm_topk_prob)),
            partial_rotary_factor=float(
                llm.get("partial_rotary_factor", defaults.partial_rotary_factor)
            ),
            rope_theta=float(llm.get("rope_theta", defaults.rope_theta)),
            rms_norm_eps=float(llm.get("rms_norm_eps", defaults.rms_norm_eps)),
            max_position_embeddings=int(
                llm.get("max_position_embeddings", defaults.max_position_embeddings)
            ),
            vocab_size=int(llm.get("vocab_size", defaults.vocab_size)),
            image_patch_token=int(
                llm.get("image_patch_token", defaults.image_patch_token)
            ),
            image_start_token=int(
                llm.get("image_start_token", defaults.image_start_token)
            ),
            image_end_token=int(llm.get("image_end_token", defaults.image_end_token)),
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
    max_position_embeddings: int = 32768
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MingImageConnectorConfig:
        defaults = cls()
        heads = int(config.get("num_attention_heads", defaults.num_attention_heads))
        hidden = int(config.get("hidden_size", defaults.hidden_size))
        return cls(
            hidden_size=hidden,
            num_hidden_layers=int(
                config.get("num_hidden_layers", defaults.num_hidden_layers)
            ),
            num_attention_heads=heads,
            num_key_value_heads=int(
                config.get("num_key_value_heads", defaults.num_key_value_heads)
            ),
            head_dim=int(config.get("head_dim", hidden // heads)),
            intermediate_size=int(
                config.get("intermediate_size", defaults.intermediate_size)
            ),
            rope_theta=float(config.get("rope_theta", defaults.rope_theta)),
            rms_norm_eps=float(config.get("rms_norm_eps", defaults.rms_norm_eps)),
            max_position_embeddings=int(
                config.get("max_position_embeddings", defaults.max_position_embeddings)
            ),
            vocab_size=int(config.get("vocab_size", defaults.vocab_size)),
        )


@dataclass(frozen=True, slots=True)
class MingImageBridgeConfig:
    query_token_scale: int = 16
    mllm_hidden: int = 2048
    connector_hidden: int = 1536
    cap_feat_dim: int = 2560
    directvlm_dim: int = 3840
    selected_hidden_states_layers: tuple[int, ...] = (5, 12, 20)
    use_identity_mlp: bool = True
    use_learnable_token_condition: bool = True
    use_vlm_directvlm_condition: bool = True

    @property
    def query_token_count(self) -> int:
        return self.query_token_scale * self.query_token_scale

    @property
    def directvlm_in(self) -> int:
        return self.mllm_hidden * len(self.selected_hidden_states_layers)

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], *, mllm_hidden: int, connector_hidden: int
    ) -> MingImageBridgeConfig:
        defaults = cls()
        scales = config.get("img_gen_scales", [defaults.query_token_scale])
        return cls(
            query_token_scale=int(scales[0]),
            mllm_hidden=mllm_hidden,
            connector_hidden=connector_hidden,
            cap_feat_dim=int(
                config.get("diffusion_c_input_dim", defaults.cap_feat_dim)
            ),
            directvlm_dim=int(
                config.get("diffusion_inner_dim", defaults.directvlm_dim)
            ),
            selected_hidden_states_layers=tuple(
                config.get(
                    "selected_hidden_states_layers",
                    defaults.selected_hidden_states_layers,
                )
            ),
            use_identity_mlp=bool(
                config.get("use_identity_mlp", defaults.use_identity_mlp)
            ),
            use_learnable_token_condition=bool(
                config.get(
                    "use_learnable_token_condition",
                    defaults.use_learnable_token_condition,
                )
            ),
            use_vlm_directvlm_condition=bool(
                config.get(
                    "use_vlm_directvlm_condition",
                    defaults.use_vlm_directvlm_condition,
                )
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
    def from_dict(cls, config: dict[str, Any]) -> MingImageVAEConfig:
        defaults = cls()
        channels = int(config.get("input_channels", config.get("in_channels", 4)))
        return cls(
            base_dim=int(config.get("base_dim", defaults.base_dim)),
            z_dim=int(config.get("z_dim", defaults.z_dim)),
            dim_mult=tuple(config.get("dim_mult", defaults.dim_mult)),
            num_res_blocks=int(config.get("num_res_blocks", defaults.num_res_blocks)),
            temperal_downsample=tuple(
                config.get("temperal_downsample", defaults.temperal_downsample)
            ),
            in_channels=channels,
            out_channels=int(config.get("out_channels", channels)),
            is_residual=bool(config.get("is_residual", defaults.is_residual)),
            scaling_factor=float(config.get("scaling_factor", defaults.scaling_factor)),
            shift_factor=float(config.get("shift_factor", defaults.shift_factor)),
        )


@dataclass(frozen=True, slots=True)
class MingImageConfig:
    dit: MingImageDiTConfig = field(default_factory=MingImageDiTConfig)
    mllm: MingImageMLLMConfig = field(default_factory=MingImageMLLMConfig)
    connector: MingImageConnectorConfig = field(
        default_factory=MingImageConnectorConfig
    )
    bridge: MingImageBridgeConfig = field(default_factory=MingImageBridgeConfig)
    vae: MingImageVAEConfig = field(default_factory=MingImageVAEConfig)
    default_steps: int = 12
    default_guidance: float = 1.0
    num_train_timesteps: int = 1000

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> MingImageConfig:
        root = Path(model_path).expanduser()
        mllm = MingImageMLLMConfig.from_dict(_load(root, "mllm/config.json"))
        connector = MingImageConnectorConfig.from_dict(
            _load(root, "connector/config.json")
        )
        scheduler = _load(root, "scheduler/scheduler_config.json")
        return cls(
            dit=MingImageDiTConfig.from_dict(_load(root, "transformer/config.json")),
            mllm=mllm,
            connector=connector,
            bridge=MingImageBridgeConfig.from_dict(
                _load(root, "mlp/config.json"),
                mllm_hidden=mllm.hidden_size,
                connector_hidden=connector.hidden_size,
            ),
            vae=MingImageVAEConfig.from_dict(_load(root, "vae/config.json")),
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
    is_dit = dit.get("_class_name") == "DiffusionTransformer"
    is_bailing = "BailingMM" in str(backbone.get("architectures", [""])[0])
    return is_dit and is_bailing


__all__ = [
    "MingImageBridgeConfig",
    "MingImageConfig",
    "MingImageConnectorConfig",
    "MingImageDiTConfig",
    "MingImageMLLMConfig",
    "MingImageVAEConfig",
    "detect_ming_image_layout",
]
