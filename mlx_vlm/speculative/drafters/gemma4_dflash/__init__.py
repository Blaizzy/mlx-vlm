from ..qwen3_dflash.config import DFlashConfig
from ..qwen3_dflash.dflash import DFlashDraftModel, DFlashKVCache


class Gemma4DFlashConfig(DFlashConfig):
    @classmethod
    def from_dict(cls, params: dict) -> "Gemma4DFlashConfig":
        flat = dict(params)
        dflash_cfg = dict(flat.get("dflash_config", None) or {})
        dflash_cfg.setdefault("mask_token_id", 4)
        flat["dflash_config"] = dflash_cfg
        return super().from_dict(flat)

    from_hf_dict = from_dict


class Gemma4DFlashDraftModel(DFlashDraftModel):
    pass


Model = Gemma4DFlashDraftModel
ModelConfig = Gemma4DFlashConfig

__all__ = [
    "Gemma4DFlashConfig",
    "Gemma4DFlashDraftModel",
    "DFlashKVCache",
    "Model",
    "ModelConfig",
]
