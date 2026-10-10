"""HTTP access to the shared typed decision API."""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator

from ..decision import DecisionCancelled
from ..decision_scheduler import DecisionQueueFull, DecisionScheduler
from ..systemone import (
    SystemOneRequest,
    SystemOneResponse,
    format_response,
    native_request,
)
from .generation import get_configured_context_limit
from .openai import _decode_input_audio_data
from .runtime import runtime

logger = logging.getLogger(__name__)


def _default_decision_model() -> Optional[str]:
    configured = os.environ.get("MLX_VLM_PRELOAD_DECISION_MODEL")
    if configured:
        return configured
    registry = runtime.model_cache
    if hasattr(registry, "for_kind"):
        return registry.for_kind("decision").get("model_path")
    return None


class DecisionRequest(BaseModel):
    model: Optional[str] = Field(default=None, min_length=1)
    state: Union[str, Dict[str, Any], list, None] = None
    questions: Dict[str, Dict[str, Any]] = Field(min_length=1)
    images: Optional[List[str]] = None
    videos: Optional[List[List[str]]] = None
    audio: Optional[str] = None

    @model_validator(mode="after")
    def _state_or_media(self):
        if self.state is None and not (self.images or self.videos or self.audio):
            raise ValueError(
                "state is required unless images, videos, or audio are given"
            )
        return self


def get_scheduler(deps):
    if runtime.decision_queue is None:
        configured = os.environ.get("MLX_VLM_DECISION_MAX_LENGTH")
        runtime.decision_queue = DecisionScheduler(
            lambda name: deps.get_cached_model(name, model_kind="decision"),
            max_length=(
                int(configured) if configured else get_configured_context_limit()
            ),
            batch_size=int(os.environ.get("MLX_VLM_DECISION_BATCH_SIZE", "4")),
            prefill_step_size=int(
                os.environ.get("MLX_VLM_DECISION_PREFILL_STEP_SIZE", "512")
            ),
            cache_bytes=int(os.environ.get("MLX_VLM_DECISION_PREFIX_CACHE_MB", "256"))
            * 1024**2,
            max_pending=int(os.environ.get("MLX_VLM_DECISION_MAX_PENDING", "64")),
        )
    return runtime.decision_queue


async def _execute(body, request, response, deps, *, systemone=False):
    endpoint = "/v1/systemone" if systemone else "/v1/decisions"
    model_id = body.get("model") or _default_decision_model()
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail="Specify a model or preload one with --decision-model",
        )
    body = {**body, "model": model_id}
    started = time.perf_counter()
    runtime.metrics.begin_request(endpoint=endpoint, model=model_id, stream=False)
    job = future = None
    try:
        native = native_request(body) if systemone else dict(body)
        if native.get("audio"):
            native["audio"] = _decode_input_audio_data({"data": native["audio"]})
        job = get_scheduler(deps).submit(
            native,
            namespace=deps.read_tenant_id(request),
            allow_single_criterion=systemone,
        )
        future = asyncio.wrap_future(job.future)
        while not future.done():
            await asyncio.wait({future}, timeout=0.05)
            if await request.is_disconnected():
                job.cancel()
                raise HTTPException(status_code=499, detail="Client disconnected")
        output = await future
        result = format_response(body, output) if systemone else output["response"]
    except asyncio.CancelledError:
        if job is not None:
            job.cancel()
        runtime.metrics.record_failure(
            endpoint=endpoint, model=model_id, stream=False, error="Request cancelled"
        )
        raise
    except Exception as error:
        runtime.metrics.record_failure(
            endpoint=endpoint, model=model_id, stream=False, error=str(error)
        )
        if isinstance(error, HTTPException):
            raise
        if isinstance(error, DecisionQueueFull):
            raise HTTPException(status_code=429, detail=str(error)) from error
        if isinstance(error, DecisionCancelled):
            raise HTTPException(status_code=503, detail=str(error)) from error
        if isinstance(error, ValueError):
            raise HTTPException(
                status_code=422 if systemone else 400, detail=str(error)
            ) from error
        logger.exception("Decision request failed")
        raise HTTPException(
            status_code=500, detail="Decision prediction failed"
        ) from error
    finally:
        if future is not None and not future.done():
            if job is not None:
                job.cancel()
            future.cancel()
    usage = result.get("usage", {})
    runtime.metrics.record_success(
        deps.build_metrics_envelope(
            endpoint=endpoint,
            model=model_id,
            stream=False,
            backend="mlx-decision-native",
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=0,
            generated_tokens=0,
            request_elapsed_s=time.perf_counter() - started,
            request_started_s=started,
            finish_reason="stop",
        )
    )
    result["model"] = model_id
    if systemone:
        from uuid import uuid4

        response.headers["x-typesafe-request-id"] = uuid4().hex
        response.headers["x-clef-cached-tokens"] = str(output["cached_tokens"])
    return result


def register_routes(app, deps):
    @app.post("/v1/decisions")
    async def create_decisions(
        body: DecisionRequest, request: Request, response: Response
    ):
        return await _execute(
            body.model_dump(exclude_none=True), request, response, deps
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def create_systemone(
        body: SystemOneRequest, request: Request, response: Response
    ):
        return await _execute(
            body.model_dump(exclude_none=True), request, response, deps, systemone=True
        )
