from pathlib import Path
from typing import Optional, Union

from transformers import AutoTokenizer

from .encoder_loader import load_encoder_model
from .utils import get_model_path, load_config

MASKED_LM_MODEL_REMAPPING = {"lfm2": "lfm2_embedding"}


def is_masked_lm_config(config: dict) -> bool:
    return any(
        str(architecture).endswith("ForMaskedLM")
        for architecture in config.get("architectures") or []
    )


def load_masked_lm_model(
    model_path: Path,
    lazy: bool = False,
    config: Optional[dict] = None,
    **kwargs,
):
    config = dict(config) if config is not None else load_config(model_path, **kwargs)
    if not is_masked_lm_config(config):
        raise ValueError("The model is not a masked-language-model checkpoint.")
    return load_encoder_model(
        model_path,
        model_remapping=MASKED_LM_MODEL_REMAPPING,
        model_class_name="MaskedLMModel",
        config=config,
        lazy=lazy,
        **kwargs,
    )


def load_masked_lm(
    path_or_hf_repo: Union[str, Path],
    *,
    revision: Optional[str] = None,
    force_download: bool = False,
    lazy: bool = False,
    strict: bool = True,
):
    model_path = get_model_path(
        path_or_hf_repo, revision=revision, force_download=force_download
    )
    model = load_masked_lm_model(model_path, lazy=lazy, strict=strict)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer
