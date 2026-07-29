"""OpenAI-compatible ``/v1/embeddings`` endpoint backed by mlx-embeddings.

The embedding logic (model loading, pooling, encoding) lives in the
``mlx_embeddings`` package; this module only adapts it to the server's HTTP
surface, auth and request metrics. ``mlx_embeddings`` is imported lazily so
importing the server never triggers a circular import (``mlx_embeddings``
itself imports ``mlx_vlm``).
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from fastapi import HTTPException
from pydantic import BaseModel

from .runtime import runtime

logger = logging.getLogger(__name__)

_EMBEDDING_MODELS: Dict[str, Tuple[object, object]] = {}


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


def _default_embedding_model() -> Optional[str]:
    return os.environ.get("MLX_VLM_PRELOAD_EMBEDDING_MODEL") or None


def load_embedding_model(model_id: str) -> Tuple[object, object]:
    """Load and cache an mlx-embeddings model, rejecting non-embedding models."""
    cached = _EMBEDDING_MODELS.get(model_id)
    if cached is not None:
        return cached

    from mlx_embeddings import classify, load
    from mlx_embeddings.utils import get_model_path, load_config

    model_path = get_model_path(model_id)
    if not classify(load_config(model_path), model_path)["is_embedding"]:
        raise ValueError(f"'{model_id}' is not an embedding model.")

    model, processor = load(model_id)
    _EMBEDDING_MODELS[model_id] = (model, processor)
    return model, processor


def _normalize_input(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not value or any(not isinstance(item, str) for item in value):
        raise ValueError("`input` must be a non-empty string or list of strings.")
    return value


def _count_prompt_tokens(processor, texts: List[str]) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return sum(len(tokenizer.encode(text)) for text in texts)
    except Exception:  # noqa: BLE001
        return 0


def register_routes(app, deps):
    @app.post("/v1/embeddings")
    async def create_embeddings(body: EmbeddingsRequest):
        model_id = body.model or _default_embedding_model()
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No embedding model specified in the request or preloaded "
                    "via --embedding-model."
                ),
            )

        runtime.metrics.begin_request(
            endpoint="/v1/embeddings", model=model_id, stream=False
        )
        try:
            texts = _normalize_input(body.input)
            model, processor = load_embedding_model(model_id)
            from mlx_embeddings import embed

            vectors = embed(model, processor, texts)
            prompt_tokens = _count_prompt_tokens(processor, texts)
        except ValueError as exc:
            runtime.metrics.record_failure(
                endpoint="/v1/embeddings", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embeddings request failed")
            runtime.metrics.record_failure(
                endpoint="/v1/embeddings", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=500, detail=str(exc))

        runtime.metrics.record_success(
            {
                "endpoint": "/v1/embeddings",
                "model": model_id,
                "stream": False,
                "backend": "mlx-embeddings",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "generated_tokens": 0,
                "request_elapsed_s": 0.0,
                "decode_elapsed_s": 0.0,
                "prefill_tok_s": 0.0,
                "decode_tok_s": 0.0,
                "finish_reason": "stop",
                "timestamp_unix": time.time(),
            }
        )
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "model": model_id,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
