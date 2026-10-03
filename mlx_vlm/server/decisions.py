"""HTTP access to the shared typed decision API."""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..decision import predict
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
    state: Union[str, Dict[str, Any], list]
    questions: Dict[str, Dict[str, Any]] = Field(min_length=1)


def register_routes(app, deps):
    @app.post("/v1/decisions")
    async def create_decisions(body: DecisionRequest):
        model_id = body.model or _default_decision_model()
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail="Specify a model or preload one with --decision-model",
            )
        endpoint = "/v1/decisions"
        started = time.perf_counter()
        runtime.metrics.begin_request(endpoint=endpoint, model=model_id, stream=False)
        try:

            def work():
                model, processor, _ = deps.get_cached_model(
                    model_id, model_kind="decision"
                )
                return predict(model, processor, body.state, body.questions)

            result = await asyncio.to_thread(work)
        except Exception as error:
            runtime.metrics.record_failure(
                endpoint=endpoint, model=model_id, stream=False, error=str(error)
            )
            if isinstance(error, HTTPException):
                raise
            if isinstance(error, ValueError):
                raise HTTPException(status_code=400, detail=str(error)) from error
            logger.exception("Decision request failed")
            raise HTTPException(
                status_code=500, detail="Decision prediction failed"
            ) from error
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
        return result
