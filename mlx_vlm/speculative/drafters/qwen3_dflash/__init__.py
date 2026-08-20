from .config import DFlashConfig as ModelConfig
from .dflash import DFlash2DraftModel, DFlashDraftModel, DFlashKVCache


def Model(config: ModelConfig):
    model_cls = (
        DFlash2DraftModel
        if "DFlash2DraftModel" in config.architectures
        else DFlashDraftModel
    )
    return model_cls(config)


__all__ = [
    "Model",
    "ModelConfig",
    "DFlash2DraftModel",
    "DFlashDraftModel",
    "DFlashKVCache",
]
