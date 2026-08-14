import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
from fastapi import HTTPException
from mlx_lm.models.base import create_causal_mask
from pydantic import BaseModel, Field

from ..prompt_utils import get_chat_template
from ..utils import get_model_path, load_image, load_video, prepare_inputs
from .runtime import runtime

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
TEXT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_MAX_LENGTH = 8192

InputItem = Union[str, Dict[str, Any]]


class RerankRequest(BaseModel):
    query: InputItem
    documents: List[InputItem] = Field(min_length=1)
    model: Optional[str] = None
    instruction: Optional[str] = None
    top_n: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = False


@dataclass(frozen=True)
class RerankItem:
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None

    @property
    def has_media(self) -> bool:
        return self.image is not None or self.video is not None


def _default_reranker_model() -> Optional[str]:
    configured = os.environ.get("MLX_VLM_PRELOAD_RERANKER_MODEL")
    if configured:
        return configured
    registry = runtime.model_cache
    if hasattr(registry, "for_kind"):
        return registry.for_kind("reranker").get("model_path")
    return None


def _url_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        url = value.get("url")
        return url.strip() if isinstance(url, str) and url.strip() else None
    return None


def normalize_item(value: InputItem, label: str) -> RerankItem:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"`{label}` must not be empty.")
        return RerankItem(text=text)
    if not isinstance(value, dict):
        raise ValueError(f"`{label}` must be text or a multimodal object.")

    text_value = value.get("text")
    text = text_value.strip() if isinstance(text_value, str) else None
    image = _url_value(value.get("image")) or _url_value(value.get("image_url"))
    video = _url_value(value.get("video")) or _url_value(value.get("video_url"))
    if not any((text, image, video)):
        raise ValueError(
            f"`{label}` must contain non-empty `text`, `image`, or `video`."
        )
    return RerankItem(text=text, image=image, video=video)


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def ensure_chat_template(processor, model_path: str) -> None:
    tokenizer = _tokenizer(processor)
    if getattr(processor, "chat_template", None) or getattr(
        tokenizer, "chat_template", None
    ):
        return
    template_path = Path(get_model_path(model_path)) / "chat_template.jinja"
    if not template_path.exists():
        raise ValueError("The reranker model does not include a chat template.")
    template = template_path.read_text(encoding="utf-8")
    processor.chat_template = template
    tokenizer.chat_template = template


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unknown_id:
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"The reranker tokenizer does not define `{token}`.")
        token_id = ids[0]
    return int(token_id)


def _input_ids(value) -> List[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) > 0 and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized reranker prompt.")
        value = value[0]
    return list(value)


def _binary_scores(model, hidden_states, attention_mask, tokenizer):
    last_from_end = mx.argmax(attention_mask[:, ::-1], axis=1)
    positions = attention_mask.shape[1] - last_from_end - 1
    pooled = hidden_states[mx.arange(hidden_states.shape[0]), positions]
    language_model = model.language_model
    if hasattr(language_model, "lm_head"):
        logits = language_model.lm_head(pooled)
    else:
        logits = language_model.model.embed_tokens.as_linear(pooled)
    yes_id = _token_id(tokenizer, "yes")
    no_id = _token_id(tokenizer, "no")
    scores = mx.sigmoid(logits[:, yes_id] - logits[:, no_id])
    mx.eval(scores)
    return [float(value) for value in scores.tolist()]


def _attention_mask(attention_mask):
    if bool(mx.all(attention_mask).item()):
        return "causal"
    valid = attention_mask.astype(mx.bool_)
    causal = create_causal_mask(attention_mask.shape[1])
    key_mask = mx.expand_dims(valid, axis=(1, 2))
    query_mask = mx.expand_dims(valid, axis=(1, 3))
    return causal[None, None, :, :] & key_mask & query_mask


def _text_messages(query: RerankItem, document: RerankItem, instruction: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<Instruct>: {instruction}\n\n<Query>: {query.text}"
                f"\n\n<Document>: {document.text}"
            ),
        },
    ]


def _padded_tokens(tokenizer, sequences: List[List[int]]):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_length = max(map(len, sequences))
    input_ids, attention_mask = [], []
    for sequence in sequences:
        padding = max_length - len(sequence)
        input_ids.append([pad_id] * padding + sequence)
        attention_mask.append([0] * padding + [1] * len(sequence))
    return mx.array(input_ids), mx.array(attention_mask)


def _score_text_batch(model, processor, query, documents, instruction):
    tokenizer = _tokenizer(processor)
    suffix = tokenizer.encode(TEXT_SUFFIX, add_special_tokens=False)
    sequences = []
    for document in documents:
        tokens = _input_ids(
            tokenizer.apply_chat_template(
                _text_messages(query, document, instruction),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        sequences.append(tokens[: DEFAULT_MAX_LENGTH - len(suffix)] + suffix)
    input_ids, attention_mask = _padded_tokens(tokenizer, sequences)
    inner = model.language_model.model
    hidden_states = inner.embed_tokens(input_ids)
    mask = _attention_mask(attention_mask)
    for layer in inner.layers:
        hidden_states = layer(hidden_states, mask, None)
    hidden_states = inner.norm(hidden_states)
    return (
        _binary_scores(model, hidden_states, attention_mask, tokenizer),
        int(attention_mask.sum().item()),
    )


def _item_content(item: RerankItem, prefix: str):
    content = [{"type": "text", "text": prefix}]
    if item.video:
        content.append({"type": "video"})
    if item.image:
        content.append({"type": "image"})
    if item.text:
        content.append({"type": "text", "text": item.text})
    return content


def _vl_messages(query: RerankItem, document: RerankItem, instruction: str):
    content = [{"type": "text", "text": f"<Instruct>: {instruction}"}]
    content.extend(_item_content(query, "<Query>:"))
    content.extend(_item_content(document, "\n<Document>:"))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def _media(items: List[RerankItem]):
    images, videos = [], []
    for item in items:
        if item.image:
            images.append(load_image(item.image))
        if item.video:
            video, _ = load_video(item.video)
            videos.append(video)
    return images or None, videos or None


def _score_vl_batch(model, processor, query, documents, instruction):
    conversations = [
        _vl_messages(query, document, instruction) for document in documents
    ]
    prompts = [
        get_chat_template(processor, messages, add_generation_prompt=True)
        for messages in conversations
    ]
    media_items = []
    for document in documents:
        media_items.extend((query, document))
    images, videos = _media(media_items)
    inputs = prepare_inputs(
        processor,
        images=images,
        videos=videos,
        prompts=prompts,
        padding_side="left",
        truncation=True,
        max_length=DEFAULT_MAX_LENGTH,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    features = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        mask=attention_mask,
    )
    hidden_states = model.language_model.model(
        input_ids,
        inputs_embeds=features.inputs_embeds,
        mask=_attention_mask(attention_mask),
        position_ids=features.position_ids,
        visual_pos_masks=features.visual_pos_masks,
        deepstack_visual_embeds=features.deepstack_visual_embeds,
    )
    return (
        _binary_scores(model, hidden_states, attention_mask, _tokenizer(processor)),
        int(attention_mask.sum().item()),
    )


def _batch_size(model_type: str) -> int:
    configured = os.environ.get("MLX_VLM_RERANK_BATCH_SIZE")
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError("MLX_VLM_RERANK_BATCH_SIZE must be an integer.") from exc
        if value < 1:
            raise ValueError("MLX_VLM_RERANK_BATCH_SIZE must be positive.")
        return value
    return 2 if model_type == "qwen3_vl" else 8


def score_documents(model, processor, config, query, documents, instruction):
    model_type = getattr(config, "model_type", None)
    if model_type not in ("qwen3", "qwen3_vl"):
        raise ValueError(f"Unsupported reranker model type: {model_type!r}.")
    if model_type == "qwen3" and (
        query.has_media or any(document.has_media for document in documents)
    ):
        raise ValueError("Qwen3 text rerankers do not support image or video inputs.")

    score_batch = _score_vl_batch if model_type == "qwen3_vl" else _score_text_batch
    scores, prompt_tokens = [], 0
    batch_size = _batch_size(model_type)
    for start in range(0, len(documents), batch_size):
        batch_scores, batch_tokens = score_batch(
            model,
            processor,
            query,
            documents[start : start + batch_size],
            instruction,
        )
        scores.extend(batch_scores)
        prompt_tokens += batch_tokens
    return scores, prompt_tokens


def register_routes(app, deps):
    get_cached_model = deps.get_cached_model
    build_metrics_envelope = deps.build_metrics_envelope

    @app.post("/v1/rerank")
    async def rerank(body: RerankRequest):
        model_id = body.model or _default_reranker_model()
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No reranker model specified in the request or preloaded "
                    "via --reranker-model."
                ),
            )
        request_start = time.perf_counter()
        runtime.metrics.begin_request(
            endpoint="/v1/rerank", model=model_id, stream=False
        )
        try:
            query = normalize_item(body.query, "query")
            documents = [
                normalize_item(document, f"documents[{index}]")
                for index, document in enumerate(body.documents)
            ]
            instruction = (body.instruction or DEFAULT_INSTRUCTION).strip()
            if not instruction:
                raise ValueError("`instruction` must not be empty.")

            def _work():
                model, processor, config = get_cached_model(
                    model_id, model_kind="reranker"
                )
                return score_documents(
                    model, processor, config, query, documents, instruction
                )

            scores, prompt_tokens = await asyncio.to_thread(_work)
        except ValueError as exc:
            runtime.metrics.record_failure(
                endpoint="/v1/rerank", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=400, detail=str(exc))
        except HTTPException:
            runtime.metrics.record_failure(
                endpoint="/v1/rerank",
                model=model_id,
                stream=False,
                error="load_failed",
            )
            raise
        except Exception as exc:
            logger.exception("Reranking request failed")
            runtime.metrics.record_failure(
                endpoint="/v1/rerank", model=model_id, stream=False, error=str(exc)
            )
            raise HTTPException(status_code=500, detail=str(exc))

        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        if body.top_n is not None:
            order = order[: body.top_n]
        results = []
        for index in order:
            result = {"index": index, "relevance_score": scores[index]}
            if body.return_documents:
                result["document"] = body.documents[index]
            results.append(result)

        runtime.metrics.record_success(
            build_metrics_envelope(
                endpoint="/v1/rerank",
                model=model_id,
                stream=False,
                backend="mlx-vlm-reranker",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                generated_tokens=0,
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start,
                finish_reason="stop",
            )
        )
        return {
            "model": model_id,
            "results": results,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
