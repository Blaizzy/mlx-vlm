from pathlib import Path

import mlx.nn as nn

from .encoder_loader import load_encoder_model

EMBEDDING_MODEL_REMAPPING = {
    "qwen3": "qwen3_embedding",
    "gemma3_text": "gemma3_embedding",
    "lfm2": "lfm2_embedding",
    "xlm-roberta": "xlm_roberta",
}


def load_embedding_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    return load_encoder_model(
        model_path,
        model_remapping=EMBEDDING_MODEL_REMAPPING,
        lazy=lazy,
        **kwargs,
    )
