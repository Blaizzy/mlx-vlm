"""Asynchronous image-to-3D jobs, including Gaussian-splat PLY."""

import io
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from ..generate.three_d import generate_3d
from .runtime import runtime

logger = logging.getLogger(__name__)


class MeshGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str | None = Field(None, min_length=1)
    input_image: str | None = None
    output_format: Literal["ply", "glb", "obj"] = "glb"
    seed: int = Field(42, ge=0, le=2**32 - 1)
    num_inference_steps: int = Field(
        50, ge=1, validation_alias=AliasChoices("num_inference_steps", "steps")
    )
    guidance_scale: float = Field(
        3.0, ge=0, validation_alias=AliasChoices("guidance_scale", "guidance")
    )
    # Idol is image-conditioned; accept only empty or placeholder prompts.
    prompt: Literal["", "generate 3d mesh"] | None = None
    negative_prompt: Literal[""] | None = None
    mode: Literal["synthetic", "real"] = "synthetic"
    num_gaussians: int = Field(262144, ge=32768, le=1048576, multiple_of=32)
    rescale_t: float = Field(3.0, gt=0)
    crop_margin: float | None = Field(None, gt=0)
    batch_cfg: bool = False
    mesh_resolution: int = Field(128, ge=32, le=256)
    mesh_threshold: float = Field(0.5, gt=0)


class MeshResponse(BaseModel):
    id: str
    object: Literal["mesh"] = "mesh"
    model: str
    status: Literal["queued", "in_progress", "completed", "failed", "deleted"]
    progress: int = 0
    created_at: int
    completed_at: int | None = None
    format: Literal["ply", "glb", "obj"]
    representation: Literal["gaussian_splat", "surface_mesh"]
    seed: int
    url: str | None = None
    file_size_bytes: int | None = None
    camera: dict | None = None
    error: dict | None = None


class MeshListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[MeshResponse]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class MeshJobs:
    """Serialize model loading and generation off the HTTP event loop.

    Records live in this process. Deleting a record does not cancel generation
    or remove exported files.
    """

    def __init__(self, deps):
        self.deps = deps
        self.jobs = {}
        self.paths = {}
        self.lock = Lock()
        self.executor = None

    def submit(self, request, image):
        job_id = f"mesh_{uuid.uuid4().hex}"
        root = Path(os.environ.get("MLX_VLM_MESH_OUTPUT_DIR", "outputs/meshes"))
        path = root.expanduser().resolve() / job_id / f"output.{request.output_format}"
        job = MeshResponse(
            id=job_id,
            model=request.model,
            status="queued",
            created_at=int(time.time()),
            format=request.output_format,
            representation=(
                "gaussian_splat" if request.output_format == "ply" else "surface_mesh"
            ),
            seed=request.seed,
        )
        with self.lock:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mlx-mesh"
                )
            self.jobs[job_id] = job
            self.paths[job_id] = path
            self.executor.submit(self._run, job_id, request, image, path)
        return job

    def _update(self, job_id, **fields):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id] = self.jobs[job_id].model_copy(update=fields)

    def _run(self, job_id, request, image, path):
        started = time.perf_counter()
        endpoint = "/v1/meshes"
        metrics = runtime.metrics
        self._update(job_id, status="in_progress")
        if metrics is not None:
            metrics.begin_request(endpoint=endpoint, model=request.model, stream=False)
        try:
            model, _, _ = self.deps.get_cached_model(
                request.model, model_kind="mesh_generation"
            )
            result = generate_3d(
                model,
                image,
                output_path=path,
                seed=request.seed,
                steps=request.num_inference_steps,
                guidance=request.guidance_scale,
                mode=request.mode,
                num_gaussians=request.num_gaussians,
                rescale_t=request.rescale_t,
                crop_margin=request.crop_margin,
                batch_cfg=request.batch_cfg,
                mesh_resolution=request.mesh_resolution,
                mesh_threshold=request.mesh_threshold,
            )
            self._update(
                job_id,
                status="completed",
                progress=100,
                completed_at=int(time.time()),
                url=f"/v1/meshes/{job_id}/content",
                file_size_bytes=path.stat().st_size,
                camera=result.camera,
            )
            if metrics is not None:
                metrics.record_success(
                    self.deps.build_metrics_envelope(
                        endpoint=endpoint,
                        model=request.model,
                        stream=False,
                        backend="mesh_generation",
                        prompt_tokens=0,
                        completion_tokens=0,
                        generated_tokens=0,
                        request_elapsed_s=time.perf_counter() - started,
                        request_started_s=started,
                        finish_reason="stop",
                    )
                )
        except Exception as error:
            message = str(getattr(error, "detail", None) or error)
            logger.exception("Mesh job %s failed", job_id)
            self._update(job_id, status="failed", error={"message": message})
            if metrics is not None:
                metrics.record_failure(
                    endpoint=endpoint, model=request.model, stream=False, error=message
                )

    def retrieve(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Mesh not found")
        return job

    def shutdown(self):
        # Drain accepted work before unloading weights or shutting down MLX.
        with self.lock:
            executor, self.executor = self.executor, None
        if executor is not None:
            executor.shutdown(wait=True)


def register_routes(app, deps):
    def jobs():
        if runtime.mesh_jobs is None:
            runtime.mesh_jobs = MeshJobs(deps)
        return runtime.mesh_jobs

    schema = MeshGenerationRequest.model_json_schema()
    form_schema = {
        **schema,
        "properties": {
            **schema["properties"],
            "image": {"type": "string", "format": "binary"},
            "url": {"type": "string"},
        },
    }

    @app.post(
        "/v1/meshes",
        response_model=MeshResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": schema},
                    "multipart/form-data": {"schema": form_schema},
                },
            }
        },
    )
    async def create_mesh(request: Request):
        if "multipart/form-data" in request.headers.get("content-type", ""):
            async with request.form() as form:
                image_keys = ("image", "image[]", "url", "url[]", "input_image")
                images = [value for key in image_keys for value in form.getlist(key)]
                if len(images) != 1:
                    raise HTTPException(422, "Exactly one image or url is required")
                image = images[0]
                if isinstance(image, UploadFile):
                    image = io.BytesIO(await image.read())
                payload = {
                    key: value for key, value in form.items() if key not in image_keys
                }
        else:
            try:
                payload = await request.json()
            except ValueError as error:
                raise HTTPException(400, "Invalid JSON body") from error
            image = payload.get("input_image") if isinstance(payload, dict) else None
        try:
            body = MeshGenerationRequest.model_validate(payload)
        except ValidationError as error:
            raise HTTPException(
                422, error.errors(include_context=False, include_input=False)
            ) from error
        if image is None or image == "":
            raise HTTPException(422, "An input image is required")
        registry = runtime.model_cache
        cached = (
            registry.for_kind("mesh_generation")
            if hasattr(registry, "for_kind")
            else {}
        )
        body.model = (
            body.model
            or cached.get("model_path")
            or os.environ.get("MLX_VLM_PRELOAD_MESH_MODEL")
        )
        if not body.model:
            raise HTTPException(400, "Specify a model or preload one with --mesh-model")
        return jobs().submit(body, image)

    @app.get("/v1/meshes", response_model=MeshListResponse)
    async def list_meshes(
        after: str | None = None,
        limit: int = Query(20, ge=1, le=100),
        order: Literal["asc", "desc"] = "desc",
    ):
        store = jobs()
        with store.lock:
            items = list(store.jobs.values())
        if order == "desc":
            items.reverse()
        if after is not None:
            ids = [item.id for item in items]
            if after not in ids:
                raise HTTPException(404, "Pagination cursor not found")
            items = items[ids.index(after) + 1 :]
        page = items[:limit]
        return MeshListResponse(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=len(items) > limit,
        )

    @app.get("/v1/meshes/{mesh_id}", response_model=MeshResponse)
    async def retrieve_mesh(mesh_id: str):
        return jobs().retrieve(mesh_id)

    @app.delete("/v1/meshes/{mesh_id}", response_model=MeshResponse)
    async def delete_mesh(mesh_id: str):
        store = jobs()
        with store.lock:
            job = store.jobs.pop(mesh_id, None)
            store.paths.pop(mesh_id, None)
        if job is None:
            raise HTTPException(404, "Mesh not found")
        return job.model_copy(update={"status": "deleted"})

    @app.get("/v1/meshes/{mesh_id}/content")
    async def download_mesh(mesh_id: str):
        store = jobs()
        job = store.retrieve(mesh_id)
        with store.lock:
            path = store.paths.get(mesh_id)
        if job.status != "completed" or path is None or not path.is_file():
            raise HTTPException(404, "Mesh output is not available")
        media_type = {
            "ply": "application/octet-stream",
            "glb": "model/gltf-binary",
            "obj": "text/plain",
        }[job.format]
        return FileResponse(path, media_type=media_type, filename=path.name)
