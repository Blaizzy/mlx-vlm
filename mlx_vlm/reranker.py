from enum import Enum
from typing import Any, Mapping


class RerankerKind(str, Enum):
    GENERATIVE_TEXT = "generative_text"
    GENERATIVE_VL = "generative_vl"
    SEQUENCE_CLASSIFIER = "sequence_classifier"


_RERANKER_KINDS = {
    "bert": RerankerKind.SEQUENCE_CLASSIFIER,
    "modernbert": RerankerKind.SEQUENCE_CLASSIFIER,
    "qwen3": RerankerKind.GENERATIVE_TEXT,
    "qwen3_vl": RerankerKind.GENERATIVE_VL,
    "xlm_roberta": RerankerKind.SEQUENCE_CLASSIFIER,
}


def reranker_model_type(config: Any) -> str:
    if isinstance(config, Mapping):
        model_type = config.get("model_type", "")
    else:
        model_type = getattr(config, "model_type", "")
    return str(model_type).lower().replace("-", "_")


def reranker_kind(config: Any) -> RerankerKind:
    model_type = reranker_model_type(config)
    kind = _RERANKER_KINDS.get(model_type)
    if kind is None:
        raise ValueError(f"Unsupported reranker model type: {model_type!r}.")
    return kind
