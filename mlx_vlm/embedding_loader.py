from pathlib import Path

import mlx.nn as nn

from .encoder_loader import load_encoder_model
from .utils import load_config

EMBEDDING_MODEL_REMAPPING = {
    "qwen3": "qwen3_embedding",
    "gemma3_text": "gemma3_embedding",
    "lfm2": "lfm2_embedding",
    "ministral3": "ministral3_embedding",
    "xlm-roberta": "xlm_roberta",
}


def load_embedding_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    model_remapping = EMBEDDING_MODEL_REMAPPING
    dense_config = model_path / "1_Dense" / "config.json"
    config_overrides = None
    if dense_config.exists():
        config_overrides = {
            "embedding_dim": load_config(dense_config.parent)["out_features"]
        }
        model_remapping = {**model_remapping, "lfm2": "lfm2_colbert"}
    return load_encoder_model(
        model_path,
        model_remapping=model_remapping,
        config_overrides=config_overrides,
        lazy=lazy,
        **kwargs,
    )
