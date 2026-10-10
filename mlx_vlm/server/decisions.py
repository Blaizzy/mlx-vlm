import asyncio
import logging
import os
import time
from uuid import uuid4

from fastapi import HTTPException, Request, Response

from ..decision import SystemOneRequest, SystemOneResponse
from ..decision_scheduler import DecisionQueueFull, DecisionScheduler
from ..models.clef.inference import DecisionCancelled
from .generation import get_configured_context_limit
from .runtime import runtime

logger = logging.getLogger(__name__)


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


def register_routes(app, deps):
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def create_decision(
        body: SystemOneRequest, response: Response, request: Request
    ):
        started = time.perf_counter()
        runtime.metrics.begin_request(
            endpoint="/v1/systemone", model=body.model, stream=False
        )

        job = None
        future = None
        try:
            job = get_scheduler(deps).submit(
                body.model_dump(exclude_none=True),
                namespace=deps.read_tenant_id(request),
            )
            future = asyncio.wrap_future(job.future)
            while not future.done():
                await asyncio.wait({future}, timeout=0.05)
                if await request.is_disconnected():
                    job.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
            output = await future
            result = output["response"]
        except asyncio.CancelledError:
            if job is not None:
                job.cancel()
            runtime.metrics.record_failure(
                endpoint="/v1/systemone",
                model=body.model,
                stream=False,
                error="Request cancelled",
            )
            raise
        except (DecisionQueueFull, DecisionCancelled) as exc:
            runtime.metrics.record_failure(
                endpoint="/v1/systemone", model=body.model, stream=False, error=str(exc)
            )
            raise HTTPException(
                status_code=429 if isinstance(exc, DecisionQueueFull) else 503,
                detail=str(exc),
            ) from exc
        except (ValueError, HTTPException) as exc:
            runtime.metrics.record_failure(
                endpoint="/v1/systemone", model=body.model, stream=False, error=str(exc)
            )
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Decision request failed")
            runtime.metrics.record_failure(
                endpoint="/v1/systemone", model=body.model, stream=False, error=str(exc)
            )
            raise HTTPException(
                status_code=500, detail="Decision inference failed"
            ) from exc
        finally:
            if future is not None and not future.done():
                if job is not None:
                    job.cancel()
                future.cancel()
        runtime.metrics.record_success(
            deps.build_metrics_envelope(
                endpoint="/v1/systemone",
                model=body.model,
                stream=False,
                backend="mlx-clef-native",
                prompt_tokens=result["usage"]["input_tokens"],
                completion_tokens=0,
                generated_tokens=0,
                request_elapsed_s=time.perf_counter() - started,
                request_started_s=started,
                finish_reason="stop",
            )
        )
        response.headers["x-typesafe-request-id"] = uuid4().hex
        response.headers["x-clef-cached-tokens"] = str(output["cached_tokens"])
        return result
