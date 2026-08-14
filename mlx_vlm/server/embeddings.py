import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from fastapi import HTTPException
from pydantic import BaseModel

from ..prompt_utils import get_chat_template
from ..utils import load_image, prepare_inputs
from .runtime import runtime

logger = logging.getLogger(__name__)

InputItem = Union[str, Dict[str, Any]]


class EmbeddingsRequest(BaseModel):
    input: Union[str, Dict[str, Any], List[InputItem]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


def _default_embedding_model() -> Optional[str]:
    return os.environ.get("MLX_VLM_PRELOAD_EMBEDDING_MODEL") or None


def _extract_image_url(item: InputItem) -> Optional[str]:
    if isinstance(item, str):
        return item if item.startswith("data:image/") else None
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type in ("image_url", "input_image"):
            url = item.get("image_url", item.get("url"))
            return url.get("url") if isinstance(url, dict) else url
        if item_type == "image":
            return item.get("image") or item.get("url")
        if "image_url" in item:
            url = item["image_url"]
            return url.get("url") if isinstance(url, dict) else url
        if "url" in item:
            return item["url"]
    return None


def _normalize_input(value: Union[InputItem, List[InputItem]]) -> List[Tuple[str, str]]:
    if isinstance(value, (str, dict)):
        value = [value]
    if not value:
        raise ValueError("`input` must be a non-empty string, image, or list.")
    items: List[Tuple[str, str]] = []
    for item in value:
        image_url = _extract_image_url(item)
        if image_url is not None:
            items.append(("image", image_url))
        elif isinstance(item, str):
            items.append(("text", item))
        else:
            raise ValueError(
                "`input` items must be strings or image objects with a URL."
            )
    return items


def _embed(model, processor, texts: List[str]):
    tok = getattr(processor, "tokenizer", processor)
    max_length = min(int(getattr(tok, "model_max_length", 512) or 512), 8192)
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
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


def _embed_image(model, processor, image_url: str):
    if getattr(processor, "image_processor", None) is None:
        raise ValueError("The loaded embedding model does not support image inputs.")
    image = load_image(image_url)
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = get_chat_template(processor, messages, add_generation_prompt=True)
    image_token_index = getattr(
        getattr(model, "config", None), "image_token_index", None
    )
    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        image_token_index=image_token_index,
    )
    if inputs.get("pixel_values") is None:
        raise ValueError("The loaded embedding model does not support image inputs.")
    out = model(**inputs)
    embeds = getattr(out, "text_embeds", None)
    if embeds is None:
        raise ValueError("The loaded embedding model does not support image inputs.")
    mx.eval(embeds)
    mask = inputs.get("attention_mask")
    tokens = int(mask.sum()) if mask is not None else 0
    return embeds[0].tolist(), tokens


def _embed_items(model, processor, items: List[Tuple[str, str]]):
    vectors: List[Any] = [None] * len(items)
    prompt_tokens = 0
    for index, (kind, value) in enumerate(items):
        if kind == "image":
            vector, tokens = _embed_image(model, processor, value)
        else:
            batch, tokens = _embed(model, processor, [value])
            vector = batch[0]
        vectors[index] = vector
        prompt_tokens += tokens
    return vectors, prompt_tokens


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
            items = _normalize_input(body.input)

            def _work():
                model, processor, *_ = get_cached_model(
                    model_id, model_kind="embedding"
                )
                if all(kind == "text" for kind, _ in items):
                    return _embed(model, processor, [value for _, value in items])
                return _embed_items(model, processor, items)

            vectors, prompt_tokens = await asyncio.to_thread(_work)
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
