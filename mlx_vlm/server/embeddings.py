import logging
import os
import time
from typing import List, Optional, Union

import mlx.core as mx
from fastapi import HTTPException
from pydantic import BaseModel

from .runtime import runtime

logger = logging.getLogger(__name__)


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


def _default_embedding_model() -> Optional[str]:
    return os.environ.get("MLX_VLM_PRELOAD_EMBEDDING_MODEL") or None


def _normalize_input(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not value or any(not isinstance(item, str) for item in value):
        raise ValueError("`input` must be a non-empty string or list of strings.")
    return value


def _embed(model, processor, texts: List[str]):
    tok = getattr(processor, "tokenizer", processor)
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    out = model(
        mx.array(enc["input_ids"]), attention_mask=mx.array(enc["attention_mask"])
    )
    embeds = getattr(out, "text_embeds", None)
    if embeds is None:
        raise ValueError("The loaded model does not produce embeddings.")
    mx.eval(embeds)
    return embeds.tolist(), int(enc["attention_mask"].sum())


def register_routes(app, deps):
    get_cached_model = deps.get_cached_model
    build_metrics_envelope = deps.build_metrics_envelope

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
        request_start = time.perf_counter()
        runtime.metrics.begin_request(
            endpoint="/v1/embeddings", model=model_id, stream=False
        )
        try:
            texts = _normalize_input(body.input)
            model, processor, *_ = get_cached_model(model_id, model_kind="embedding")
            vectors, prompt_tokens = _embed(model, processor, texts)
        except ValueError as exc:
            runtime.metrics.record_failure(
                endpoint="/v1/embeddings", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=400, detail=str(exc))
        except HTTPException:
            runtime.metrics.record_failure(
                endpoint="/v1/embeddings",
                model=model_id,
                stream=False,
                error="load_failed",
            )
            raise
        except Exception as exc:
            logger.exception("Embeddings request failed")
            runtime.metrics.record_failure(
                endpoint="/v1/embeddings", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=500, detail=str(exc))

        runtime.metrics.record_success(
            build_metrics_envelope(
                endpoint="/v1/embeddings",
                model=model_id,
                stream=False,
                backend="mlx-embeddings-native",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                generated_tokens=0,
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start,
                finish_reason="stop",
            )
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
