import inspect
from dataclasses import dataclass, field
from typing import List

from ....models.base import BaseModelConfig


@dataclass
class DFlashConfig(BaseModelConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])
    num_target_layers: int = 32

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        """Build from a raw HF config dict. Flattens the nested
        ``dflash_config`` sub-dict (``mask_token_id``, ``target_layer_ids``)
        so :func:`mlx_vlm.utils.load_model`'s generic
        ``ModelConfig.from_dict(config)`` call works unmodified.
        """
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        if "mask_token_id" in dflash_cfg:
            flat["mask_token_id"] = dflash_cfg["mask_token_id"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    # Legacy alias.
    from_hf_dict = from_dict
