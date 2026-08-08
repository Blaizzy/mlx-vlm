from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from mlx_vlm.generate.video_generation import (
    VideoGenerationModel,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReference,
    VideoWorkflow,
)

from .pipeline import MiniMaxH3GenerationRequest, MiniMaxH3Pipeline
from .references import MiniMaxH3Reference
from .weights import load_pipeline


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _can_load(model: str) -> bool:
    path = Path(model).expanduser()
    if not path.exists():
        return "minimax-h3" in model.lower().replace("_", "-")
    manifest = _load_json(path / "h3_manifest.json")
    if manifest is not None and manifest.get("format") == "mlx-vlm-minimax-h3":
        return True
    for filename in ("model_index.json", "modular_model_index.json"):
        metadata = _load_json(path / filename)
        if str((metadata or {}).get("_class_name") or "").startswith("MiniMaxH3"):
            return True
    return False


def _reference_to_h3(reference: VideoReference) -> MiniMaxH3Reference:
    if reference.kind == "image":
        return MiniMaxH3Reference(image=reference.path)
    if reference.kind == "video":
        return MiniMaxH3Reference(video=reference.path)
    return MiniMaxH3Reference(audio=reference.path)


@dataclass(slots=True)
class MiniMaxH3VideoGenerationModel(VideoGenerationModel):
    is_video_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "minimax_h3"
    pipeline: MiniMaxH3Pipeline
    model_id: str
    workflow: VideoWorkflow
    family: str = "minimax_h3"

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return _can_load(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "MiniMaxAI/MiniMax-H3",
        **kwargs: Any,
    ) -> "MiniMaxH3VideoGenerationModel":
        workflow = kwargs.pop("workflow", None)
        model_path = kwargs.pop("model_path", None)
        pipeline = load_pipeline(
            model_path or model,
            workflow=workflow,
            text_only=kwargs.pop("text_only", None),
            revision=kwargs.pop("revision", None),
            local_dir=kwargs.pop("local_dir", None),
            token=kwargs.pop("token", None),
            force_download=kwargs.pop("force_download", False),
            max_workers=kwargs.pop("max_workers", 16),
        )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MiniMax-H3 load arguments: {names}")
        if workflow is None:
            workflow = "ref2va" if pipeline.partition == "ref2va" else "t2va"
        return cls(pipeline=pipeline, model_id=str(model), workflow=workflow)

    def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        seed = 0 if request.seed is None else request.seed
        references = [_reference_to_h3(item) for item in request.references]
        result = self.pipeline.generate(
            MiniMaxH3GenerationRequest(
                prompt=request.prompt,
                image=request.image,
                last_image=request.last_image,
                references=references or None,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                num_inference_steps=request.steps,
                seed=seed,
                latents=request.extra.get("latents"),
                audio_latents=request.extra.get("audio_latents"),
                progress_callback=request.progress_callback,
                cache_adaln=request.extra.get("cache_adaln", True),
                drop_adaln_weights=request.extra.get("drop_adaln_weights", True),
            )
        )
        frames = result.video
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        frames = mx.clip(frames * 255.0, 0.0, 255.0).astype(mx.uint8)
        audio = result.audio
        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio[0]
        metadata = dict(result.metadata)
        metadata["model_id"] = self.model_id
        return VideoGenerationResult(
            frames=frames,
            audio=audio,
            fps=float(result.fps),
            sampling_rate=result.sampling_rate,
            seed=seed,
            width=int(result.metadata.get("width", frames.shape[2])),
            height=int(result.metadata.get("height", frames.shape[1])),
            num_frames=int(result.metadata.get("num_frames", frames.shape[0])),
            steps=request.steps,
            model=self.model_id,
            family=self.family,
            workflow=self.workflow,
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )


def load(
    model: str = "MiniMaxAI/MiniMax-H3",
    **kwargs: Any,
) -> MiniMaxH3VideoGenerationModel:
    return MiniMaxH3VideoGenerationModel.from_model_id(model, **kwargs)


__all__ = ["MiniMaxH3VideoGenerationModel", "load"]
