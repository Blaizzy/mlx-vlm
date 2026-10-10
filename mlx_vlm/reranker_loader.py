from pathlib import Path
from typing import Optional, Union

from .encoder_loader import load_encoder_model
from .reranker import RerankerKind, reranker_kind
from .utils import get_model_path, load, load_config, load_processor

SEQUENCE_CLASSIFIER_MODEL_REMAPPING = {"xlm-roberta": "xlm_roberta"}


def is_sequence_classifier_config(config: dict) -> bool:
    return any(
        str(architecture).endswith("ForSequenceClassification")
        for architecture in config.get("architectures") or []
    )


def _num_labels(config: dict) -> int:
    value = config.get("num_labels")
    if value is None and isinstance(config.get("id2label"), dict):
        value = len(config["id2label"])
    return int(value or 1)


def load_sequence_classification_model(
    model_path: Path,
    lazy: bool = False,
    config: Optional[dict] = None,
    **kwargs,
):
    config = dict(config) if config is not None else load_config(model_path, **kwargs)
    if not is_sequence_classifier_config(config):
        raise ValueError("The model is not a sequence-classification checkpoint.")

    if reranker_kind(config) != RerankerKind.SEQUENCE_CLASSIFIER:
        raise ValueError(
            "The sequence-classification checkpoint does not use a supported "
            "reranker backbone."
        )

    num_labels = _num_labels(config)
    if num_labels != 1:
        raise ValueError(
            "Reranker sequence classifiers must expose exactly one output label."
        )

    return load_encoder_model(
        model_path,
        model_remapping=SEQUENCE_CLASSIFIER_MODEL_REMAPPING,
        model_class_name="SequenceClassificationModel",
        config=config,
        config_overrides={"num_labels": num_labels},
        lazy=lazy,
        **kwargs,
    )


def load_reranker(
    path_or_hf_repo: Union[str, Path],
    lazy: bool = False,
    revision: Optional[str] = None,
    strict: bool = True,
    **kwargs,
):
    model_path = get_model_path(
        path_or_hf_repo,
        revision=revision,
        force_download=kwargs.get("force_download", False),
    )
    config = load_config(model_path, **kwargs)
    kind = reranker_kind(config)
    if kind != RerankerKind.SEQUENCE_CLASSIFIER:
        return load(model_path, lazy=lazy, strict=strict, **kwargs)
    if not is_sequence_classifier_config(config):
        raise ValueError(
            "BERT-family rerankers must be sequence-classification checkpoints."
        )

    model = load_sequence_classification_model(
        model_path, lazy=lazy, strict=strict, config=config, **kwargs
    )
    processor = load_processor(model_path, add_detokenizer=False, **kwargs)
    return model, processor
