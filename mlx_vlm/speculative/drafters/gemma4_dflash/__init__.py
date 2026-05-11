from ..qwen3_dflash.config import DFlashConfig
from ..qwen3_dflash.dflash import DFlashDraftModel, DFlashKVCache


class Gemma4DFlashConfig(DFlashConfig):
    @classmethod
    def from_dict(cls, params: dict) -> "Gemma4DFlashConfig":
        config = super().from_dict(params)
        if config.num_target_layers == 30:
            # The 26B-A4B DFlash checkpoint publishes one-based target layer ids
            # for the MLX Gemma4 capture path.
            config.target_layer_ids = [
                max(0, layer_id - 1) for layer_id in config.target_layer_ids
            ]
        return config

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
