from __future__ import annotations

import importlib
import json
import random
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Protocol, Sequence

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from .image import parse_size

DEFAULT_VIDEO_FORMAT = "mp4"
DEFAULT_VIDEO_STEPS = 30

VideoReferenceKind = Literal["image", "video", "audio"]
VideoWorkflow = Literal["t2va", "fl2va", "ref2va"]
VideoProgressCallback = Callable[[str, int, int, int | None], None]


@dataclass(frozen=True, slots=True)
class VideoReference:
    kind: VideoReferenceKind
    path: str | Path

    def __post_init__(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError(
                f"Video reference kind must be image, video, or audio, got {self.kind!r}"
            )


@dataclass(slots=True)
class VideoGenerationRequest:
    prompt: str
    seed: int | None = None
    steps: int = DEFAULT_VIDEO_STEPS
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    image: str | Path | None = None
    last_image: str | Path | None = None
    references: tuple[VideoReference, ...] = ()
    progress_callback: VideoProgressCallback | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoGenerationResult:
    frames: mx.array
    audio: mx.array | None
    fps: float
    sampling_rate: int | None
    seed: int
    width: int
    height: int
    num_frames: int
    steps: int
    model: str
    family: str
    workflow: VideoWorkflow
    peak_memory: float = 0.0
    path: Path | None = None
    output_format: Literal["mp4"] = DEFAULT_VIDEO_FORMAT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mx.eval(self.frames)
        if self.audio is not None:
            mx.eval(self.audio)

    @property
    def video(self) -> mx.array:
        return self.frames

    def save(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Video generation currently writes MP4 output")
        self.path = save_video(
            self.frames,
            output_path,
            fps=self.fps,
            audio=self.audio,
            sampling_rate=self.sampling_rate,
        )
        return self.path


class VideoGenerationModel(Protocol):
    is_video_generation_model: ClassVar[bool]
    model_type: ClassVar[str]
    model_id: str
    family: str
    workflow: VideoWorkflow

    @classmethod
    def supports_model(cls, model: str) -> bool: ...

    @classmethod
    def from_model_id(cls, model: str, **kwargs: Any) -> "VideoGenerationModel": ...

    def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult: ...


class _VideoProgressBar:
    _LABELS = {
        "load": "Loading video model",
        "prepare": "Preparing conditioning",
        "cache_adaln": "Caching AdaLN",
        "denoise": "Denoising video",
        "decode": "Decoding audio/video",
        "decoded": "Audio/video decoded",
        "encode": "Encoding MP4",
        "complete": "Video complete",
    }

    def __init__(
        self,
        *,
        steps: int,
        num_frames: int | None,
        disable: bool,
    ) -> None:
        self._num_frames = num_frames
        self._started_at: float | None = None
        self._bar = tqdm(
            total=steps,
            desc=self._LABELS["load"],
            unit="step",
            disable=disable,
            dynamic_ncols=True,
        )

    def __enter__(self) -> "_VideoProgressBar":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._bar.close()

    def __call__(
        self,
        phase: str,
        completed: int,
        total: int,
        num_frames: int | None,
    ) -> None:
        if phase == "prepare" and self._started_at is None:
            self._started_at = time.perf_counter()
        if num_frames is not None:
            self._num_frames = num_frames
        if total > 0 and self._bar.total != total:
            self._bar.total = total

        self._bar.set_description_str(self._LABELS.get(phase, phase.capitalize()))
        if phase == "cache_adaln":
            self._bar.set_postfix(
                {"cache": f"{completed}/{total}"},
                refresh=False,
            )
        elif phase == "denoise":
            if completed == 0:
                self._bar.set_postfix({}, refresh=False)
            target = max(0, min(completed, total))
            self._bar.update(max(0, target - self._bar.n))
        elif phase in {"decode", "decoded", "encode", "complete"}:
            self._bar.update(max(0, total - self._bar.n))

        elapsed = (
            time.perf_counter() - self._started_at
            if self._started_at is not None
            else 0.0
        )
        if elapsed > 0 and self._num_frames and total > 0:
            if phase == "denoise":
                frame_equivalents = self._num_frames * completed / total
            elif phase in {"decode", "decoded", "encode", "complete"}:
                frame_equivalents = self._num_frames
            else:
                frame_equivalents = 0.0
            if frame_equivalents > 0:
                self._bar.set_postfix(
                    {"frames/s": f"{frame_equivalents / elapsed:.2f}"},
                    refresh=False,
                )
        self._bar.refresh()


def _normalize_frames(frames: mx.array | np.ndarray) -> np.ndarray:
    array = np.array(frames)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            "Video frames must have shape (frames, height, width, 3), "
            f"got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("Video frames contain non-finite values")
        if array.size and (array.min() < 0.0 or array.max() > 1.0):
            raise ValueError("Floating video frames must be in the range [0, 1]")
        array = np.rint(array * 255.0)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _normalize_audio(audio: mx.array | np.ndarray) -> np.ndarray:
    array = np.array(audio, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not 1 <= array.shape[0] <= 8:
        raise ValueError(
            "Audio must have channels-first shape (channels, samples), "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Audio contains non-finite values")
    return np.ascontiguousarray(np.clip(array, -1.0, 1.0))


def _write_pcm_wav(audio: np.ndarray, path: Path, sampling_rate: int) -> None:
    samples = np.rint(audio.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(audio.shape[0])
        output.setsampwidth(2)
        output.setframerate(sampling_rate)
        output.writeframes(samples.tobytes())


def _ffmpeg_error(stderr: Any) -> str:
    stderr.seek(0)
    message = stderr.read().decode("utf-8", errors="replace").strip()
    return message or "unknown FFmpeg error"


def save_video(
    frames: mx.array | np.ndarray,
    output_path: str | Path,
    *,
    fps: float,
    audio: mx.array | np.ndarray | None = None,
    sampling_rate: int | None = None,
) -> Path:
    """Encode RGB frames and optional channels-first audio into an MP4 file."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    video = _normalize_frames(frames)
    if video.shape[0] == 0:
        raise ValueError("Video must contain at least one frame")
    height, width = video.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError(
            f"H.264 yuv420p output requires even dimensions, got {width}x{height}"
        )

    waveform = None
    if audio is not None:
        if sampling_rate is None or sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive when audio is provided")
        waveform = _normalize_audio(audio)
        if waveform.shape[1] == 0:
            raise ValueError("Audio must contain at least one sample")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to encode generated video output")

    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video generation currently writes MP4 output")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mlx-vlm-video-") as temporary:
        temporary_root = Path(temporary)
        encoded_path = temporary_root / "output.mp4"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:g}",
            "-i",
            "pipe:0",
        ]
        if waveform is not None:
            audio_path = temporary_root / "audio.wav"
            _write_pcm_wav(waveform, audio_path, sampling_rate)
            command.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"])
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if waveform is not None:
            command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        command.extend(["-movflags", "+faststart", str(encoded_path)])

        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
            try:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg input pipe was not created")
                for frame in video:
                    process.stdin.write(memoryview(frame).cast("B"))
                process.stdin.close()
            except BrokenPipeError:
                pass
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"FFmpeg failed: {_ffmpeg_error(stderr)}")

        shutil.move(str(encoded_path), destination)
    return destination


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _model_type_from_id(model: str) -> str | None:
    normalized = model.strip().lower().rstrip("/").replace("_", "-")
    name = normalized.rsplit("/", 1)[-1]
    if "minimax-h3" in name:
        return "minimax_h3"
    return None


def _local_model_type(model: str) -> str | None:
    root = Path(model).expanduser()
    if not root.exists():
        return None
    manifest = _load_json(root / "h3_manifest.json")
    if manifest is not None and manifest.get("format") == "mlx-vlm-minimax-h3":
        return "minimax_h3"
    for filename in ("model_index.json", "modular_model_index.json"):
        metadata = _load_json(root / filename)
        class_name = str((metadata or {}).get("_class_name") or "")
        if class_name.startswith("MiniMaxH3"):
            return "minimax_h3"
    return None


@lru_cache(maxsize=32)
def _video_model_class_for_type(model_type: str | None) -> type[Any] | None:
    if model_type is None:
        return None
    package = f"mlx_vlm.models.{model_type}"
    module_name = f"{package}.model"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {package, module_name}:
            return None
        raise
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and getattr(value, "is_video_generation_model", False)
            and getattr(value, "model_type", None) == model_type
        ):
            return value
    return None


def video_generation_model_class(model: str | None) -> type[Any] | None:
    if model is None:
        return None
    for model_type in (_local_model_type(model), _model_type_from_id(model)):
        model_class = _video_model_class_for_type(model_type)
        if model_class is not None and model_class.supports_model(model):
            return model_class
    return None


def is_video_generation_model(model: str | None) -> bool:
    return video_generation_model_class(model) is not None


def load_video_generation_model(
    model: str | None,
    **kwargs: Any,
) -> VideoGenerationModel:
    if model is None:
        raise ValueError("Video generation model must be specified")
    model_class = video_generation_model_class(model)
    if model_class is None:
        raise ValueError(f"Video generation model {model} is not supported")
    return model_class.from_model_id(model, **kwargs)


def _request_from_prompt(prompt: str, **kwargs: Any) -> VideoGenerationRequest:
    supported = {
        "seed",
        "steps",
        "width",
        "height",
        "num_frames",
        "image",
        "last_image",
        "references",
        "progress_callback",
        "extra",
    }
    request_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    extra = dict(request_kwargs.pop("extra", {}) or {})
    for key, value in kwargs.items():
        if key not in supported and value is not None:
            extra[key] = value
    request_kwargs["extra"] = extra
    return VideoGenerationRequest(prompt=prompt, **request_kwargs)


def generate_video(
    model: VideoGenerationModel | Any,
    request: VideoGenerationRequest | str,
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> VideoGenerationResult:
    if isinstance(request, str):
        request = _request_from_prompt(request, **kwargs)
    elif kwargs:
        extra = dict(request.extra)
        extra.update({key: value for key, value in kwargs.items() if value is not None})
        request = replace(request, extra=extra)
    if request.seed is None:
        request = replace(request, seed=random.randrange(2**32))
    result = model.generate(request)
    progress_steps = int(result.metadata.get("num_denoising_steps", request.steps))
    if output_path is not None:
        if request.progress_callback is not None:
            request.progress_callback(
                "encode", progress_steps, progress_steps, result.num_frames
            )
        result.save(output_path)
    if request.progress_callback is not None:
        request.progress_callback(
            "complete", progress_steps, progress_steps, result.num_frames
        )
    return result


def _parse_reference(value: str) -> VideoReference:
    try:
        kind, path = value.split("=", 1)
    except ValueError as exc:
        raise ValueError(f"Reference must use KIND=PATH syntax, got {value!r}") from exc
    kind = kind.strip().lower()
    path = path.strip()
    if not path:
        raise ValueError(f"Reference path must not be empty in {value!r}")
    return VideoReference(kind=kind, path=path)  # type: ignore[arg-type]


def _prompt_text(prompt: str | Sequence[str]) -> str:
    if isinstance(prompt, str):
        return prompt
    return " ".join(str(part) for part in prompt)


def _select_workflow(
    requested: str | None,
    *,
    has_keyframes: bool,
    has_references: bool,
) -> VideoWorkflow:
    inferred: VideoWorkflow
    if has_references:
        inferred = "ref2va"
    elif has_keyframes:
        inferred = "fl2va"
    else:
        inferred = "t2va"
    if requested is None:
        return inferred
    if requested not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unsupported video workflow {requested!r}")
    if requested != inferred:
        raise ValueError(
            f"--workflow {requested} does not match the supplied conditioning; "
            f"use {inferred}"
        )
    return requested  # type: ignore[return-value]


def run_video_generation_cli(args: Any) -> None:
    prompt = _prompt_text(args.prompt)
    if not prompt:
        raise ValueError("--prompt must not be empty for video generation")

    image_paths = list(getattr(args, "image", None) or ())
    if len(image_paths) > 1:
        raise ValueError("Video FL2VA accepts at most one --image first frame")
    first_image = image_paths[0] if image_paths else None
    last_image = getattr(args, "last_image", None)
    references = tuple(
        _parse_reference(value) for value in (getattr(args, "reference", None) or ())
    )
    has_keyframes = first_image is not None or last_image is not None
    if has_keyframes and references:
        raise ValueError("FL2VA keyframes and Ref2VA references cannot be combined")
    if getattr(args, "audio", None) or getattr(args, "video", None):
        raise ValueError(
            "Video generation references must use repeatable --reference KIND=PATH "
            "arguments so their order is preserved"
        )
    workflow = _select_workflow(
        getattr(args, "workflow", None),
        has_keyframes=has_keyframes,
        has_references=bool(references),
    )

    width = height = None
    if getattr(args, "size", None) is not None:
        width, height = parse_size(args.size)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    steps = getattr(args, "steps", None)
    if steps is None:
        steps = DEFAULT_VIDEO_STEPS
    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else Path("outputs") / f"video-{seed}.mp4"
    )

    requested_frames = getattr(args, "num_frames", None)
    with _VideoProgressBar(
        steps=steps,
        num_frames=requested_frames,
        disable=not getattr(args, "verbose", False),
    ) as progress:
        progress("load", 0, steps, requested_frames)
        model = load_video_generation_model(
            args.model,
            workflow=workflow,
            revision=getattr(args, "revision", None),
            force_download=getattr(args, "force_download", False),
        )
        request = VideoGenerationRequest(
            prompt=prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            num_frames=requested_frames,
            image=first_image,
            last_image=last_image,
            references=references,
            progress_callback=progress,
            extra=dict(getattr(args, "gen_kwargs", {}) or {}),
        )
        generation_started_at = time.perf_counter()
        result = generate_video(model, request, output_path=output_path)
        generation_elapsed = time.perf_counter() - generation_started_at
    generation_fps = result.num_frames / max(generation_elapsed, 1e-9)
    print(
        f"Saved {result.path} seed={result.seed} size={result.width}x{result.height} "
        f"frames={result.num_frames} fps={result.fps:g} steps={result.steps} "
        f"workflow={result.workflow} generation_fps={generation_fps:.2f}"
    )


__all__ = [
    "DEFAULT_VIDEO_FORMAT",
    "DEFAULT_VIDEO_STEPS",
    "VideoGenerationModel",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoProgressCallback",
    "VideoReference",
    "VideoReferenceKind",
    "VideoWorkflow",
    "generate_video",
    "is_video_generation_model",
    "load_video_generation_model",
    "run_video_generation_cli",
    "save_video",
    "video_generation_model_class",
]
