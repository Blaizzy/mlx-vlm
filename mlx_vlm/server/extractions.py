"""HTTP access to the shared extraction API."""

import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..extraction import METADATA, describe_outputs, extract
from .runtime import runtime

logger = logging.getLogger(__name__)


def _default_extraction_model() -> Optional[str]:
    configured = os.environ.get("MLX_VLM_PRELOAD_EXTRACTION_MODEL")
    if configured:
        return configured
    registry = runtime.model_cache
    if hasattr(registry, "for_kind"):
        return registry.for_kind("extraction").get("model_path")
    return None


class ExtractionRequest(BaseModel):
    model: Optional[str] = Field(default=None, min_length=1)
    image: Optional[Union[str, List[str]]] = None
    video: Optional[str] = None
    task: Optional[str] = None
    prompt: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    max_frames: int = -1


def _read_inputs(body: ExtractionRequest):
    """Turn the request's image or video reference into one array."""
    from ..utils import load_image

    if body.video:
        from ..models.video_depth_anything.generate import read_video_frames

        frames = read_video_frames(body.video, max_len=body.max_frames)
        return frames[0] if isinstance(frames, tuple) else frames

    sources = [body.image] if isinstance(body.image, str) else list(body.image or [])
    if not sources:
        raise ValueError("Provide an image or a video")
    frames = [np.asarray(load_image(source).convert("RGB")) for source in sources]
    shapes = {frame.shape for frame in frames}
    if len(shapes) > 1:
        raise ValueError(f"Images must share a shape, got {sorted(shapes)}")
    return frames[0] if len(frames) == 1 else np.stack(frames)


def _encode_arrays(outputs) -> str:
    """Pack the named arrays as base64 safetensors."""
    from safetensors.numpy import save

    arrays = {name: value for name, value in outputs.items() if name != METADATA}
    return base64.b64encode(save(arrays)).decode("ascii")


def register_routes(app, deps):
    @app.post("/v1/extractions")
    async def create_extractions(body: ExtractionRequest):
        model_id = body.model or _default_extraction_model()
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail="Specify a model or preload one with --extraction-model",
            )
        endpoint = "/v1/extractions"
        started = time.perf_counter()
        runtime.metrics.begin_request(endpoint=endpoint, model=model_id, stream=False)
        try:
            inputs = _read_inputs(body)

            def work():
                model, processor, _ = deps.get_cached_model(
                    model_id, model_kind="extraction"
                )
                settings = dict(body.settings)
                if body.prompt is not None:
                    settings["text_prompt"] = body.prompt
                outputs = extract(model, processor, inputs, task=body.task, **settings)
                # Evaluate here: MLX arrays belong to this thread's stream, and
                # touching them once the result is back on the event loop aborts.
                return model, {
                    name: (
                        value
                        if name == METADATA
                        else np.ascontiguousarray(np.asarray(value))
                    )
                    for name, value in outputs.items()
                }

            model, outputs = await asyncio.to_thread(work)
        except Exception as error:
            runtime.metrics.record_failure(
                endpoint=endpoint, model=model_id, stream=False, error=str(error)
            )
            if isinstance(error, HTTPException):
                raise
            if isinstance(error, (ValueError, OSError)):
                raise HTTPException(status_code=400, detail=str(error)) from error
            if isinstance(error, TypeError) and "unexpected keyword" in str(error):
                raise HTTPException(status_code=400, detail=str(error)) from error
            logger.exception("Extraction request failed")
            raise HTTPException(status_code=500, detail="Extraction failed") from error

        task = body.task or getattr(model, "extraction_types", ("",))[0]
        runtime.metrics.record_success(
            deps.build_metrics_envelope(
                endpoint=endpoint,
                model=model_id,
                stream=False,
                backend="mlx-extraction-native",
                prompt_tokens=0,
                completion_tokens=0,
                generated_tokens=0,
                request_elapsed_s=time.perf_counter() - started,
                request_started_s=started,
                finish_reason="stop",
            )
        )
        payload = {"model": model_id, "task": task, **describe_outputs(outputs)}
        payload["arrays_b64"] = _encode_arrays(outputs)
        return payload
