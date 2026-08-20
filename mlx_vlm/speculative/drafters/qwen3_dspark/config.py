import inspect
from dataclasses import dataclass

from ..qwen3_dflash.config import DFlashConfig


@dataclass
class DSparkConfig(DFlashConfig):
    """Qwen3 DSpark checkpoint configuration.

    SpecForge stores DSpark as a Qwen3 DFlash backbone plus Markov and
    confidence heads. ``block_size`` is the trained draft width; the runtime
    block size includes the target bonus token, matching mlx-vlm's other
    speculative backends.
    """

    model_type: str = "dspark"
    logits_start: int = 0
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    runtime_block_size: int | None = None

    @classmethod
    def from_dict(cls, params: dict) -> "DSparkConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        if dflash_cfg.get("projector_type") != "dspark":
            raise ValueError(
                "DSpark checkpoints require " "dflash_config.projector_type='dspark'."
            )

        for key in ("mask_token_id", "target_layer_ids"):
            if key in dflash_cfg:
                flat[key] = dflash_cfg[key]

        # SpecForge Qwen3 drafters build RoPE from ``rope_parameters``. In
        # particular, Qwen3.8 uses YaRN even at short positions through mscale.
        rope = flat.get("rope_parameters") or flat.get("rope_scaling")
        if rope:
            flat["rope_scaling"] = dict(rope)
            flat["rope_theta"] = rope.get(
                "rope_theta", flat.get("rope_theta", cls.rope_theta)
            )

        flat["model_type"] = "dspark"
        flat.setdefault("logits_start", 0)
        sig = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in sig})

    from_hf_dict = from_dict
