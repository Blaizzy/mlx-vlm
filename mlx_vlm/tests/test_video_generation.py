import importlib
import json
import shutil
import subprocess
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate import (
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReference,
    is_video_generation_model,
    video_generation_model_class,
)
from mlx_vlm.models.minimax_h3.model import MiniMaxH3VideoGenerationModel

dispatch_module = importlib.import_module("mlx_vlm.generate.dispatch")
video_module = importlib.import_module("mlx_vlm.generate.video_generation")


def _result(path=None):
    return VideoGenerationResult(
        frames=mx.zeros((2, 32, 64, 3), mx.uint8),
        audio=mx.zeros((2, 2667), mx.float32),
        fps=24.0,
        sampling_rate=32000,
        seed=7,
        width=64,
        height=32,
        num_frames=2,
        steps=2,
        model="MiniMaxAI/MiniMax-H3",
        family="minimax_h3",
        workflow="ref2va",
        path=path,
    )


def test_video_model_discovery_uses_h3_metadata(tmp_path):
    (tmp_path / "h3_manifest.json").write_text(
        json.dumps({"format": "mlx-vlm-minimax-h3", "partition": "fl2va"})
    )

    assert video_generation_model_class(str(tmp_path)) is MiniMaxH3VideoGenerationModel
    assert is_video_generation_model("MiniMaxAI/MiniMax-H3")
    assert not is_video_generation_model("example/not-a-video-model")


def test_h3_video_adapter_maps_ordered_references_and_outputs():
    calls = []
    progress = Mock()

    class FakePipeline:
        partition = "ref2va"

        def generate(self, request):
            calls.append(request)
            return SimpleNamespace(
                video=mx.full((1, 2, 32, 64, 3), 0.5, mx.float32),
                audio=mx.zeros((1, 2, 2667), mx.float32),
                fps=24,
                sampling_rate=32000,
                metadata={"width": 64, "height": 32, "num_frames": 2},
            )

    model = MiniMaxH3VideoGenerationModel(
        pipeline=FakePipeline(),
        model_id="synthetic-h3",
        workflow="ref2va",
    )
    result = model.generate(
        VideoGenerationRequest(
            prompt="synthetic",
            seed=7,
            steps=2,
            num_frames=2,
            progress_callback=progress,
            references=(
                VideoReference("image", "first.png"),
                VideoReference("video", "motion.mp4"),
                VideoReference("audio", "sound.wav"),
            ),
        )
    )

    assert [reference.kind for reference in calls[0].references] == [
        "image",
        "video",
        "audio",
    ]
    assert result.frames.shape == (2, 32, 64, 3)
    assert result.frames.dtype == mx.uint8
    assert result.audio.shape == (2, 2667)
    assert result.workflow == "ref2va"
    assert calls[0].progress_callback is progress


def test_video_generation_cli_preserves_reference_order(tmp_path, capsys):
    output_path = tmp_path / "generated.mp4"
    args = Namespace(
        model="MiniMaxAI/MiniMax-H3",
        prompt=["A", "short", "film"],
        image=None,
        last_image=None,
        reference=[
            "image=character.png",
            "video=motion.mp4",
            "audio=voice.wav",
        ],
        audio=None,
        video=None,
        workflow=None,
        size="64x32",
        seed=7,
        steps=2,
        num_frames=124,
        output=str(output_path),
        revision="test-revision",
        force_download=False,
        gen_kwargs={"test": True},
        verbose=True,
    )
    model = SimpleNamespace()
    result = _result(output_path)

    with (
        patch.object(video_module, "_VideoProgressBar") as mock_progress,
        patch.object(
            video_module, "load_video_generation_model", return_value=model
        ) as mock_load,
        patch.object(
            video_module, "generate_video", return_value=result
        ) as mock_generate,
    ):
        video_module.run_video_generation_cli(args)

    assert mock_load.call_args.kwargs["workflow"] == "ref2va"
    request = mock_generate.call_args.args[1]
    assert request.prompt == "A short film"
    assert [(item.kind, str(item.path)) for item in request.references] == [
        ("image", "character.png"),
        ("video", "motion.mp4"),
        ("audio", "voice.wav"),
    ]
    assert request.width == 64
    assert request.height == 32
    assert request.num_frames == 124
    progress = mock_progress.return_value.__enter__.return_value
    mock_progress.assert_called_once_with(steps=2, num_frames=124, disable=False)
    progress.assert_any_call("load", 0, 2, 124)
    assert request.progress_callback is progress
    assert mock_generate.call_args.kwargs["output_path"] == output_path
    output = capsys.readouterr().out
    assert "workflow=ref2va" in output
    assert "generation_fps=" in output


@pytest.mark.parametrize(
    ("image", "last_image", "expected_workflow"),
    [
        (None, None, "t2va"),
        (["first.png"], None, "fl2va"),
        (None, "last.png", "fl2va"),
    ],
)
def test_video_generation_cli_infers_keyframe_workflow(
    tmp_path, image, last_image, expected_workflow
):
    args = Namespace(
        model="MiniMaxAI/MiniMax-H3",
        prompt=["synthetic"],
        image=image,
        last_image=last_image,
        reference=None,
        audio=None,
        video=None,
        workflow=None,
        size=None,
        seed=7,
        steps=None,
        num_frames=None,
        output=str(tmp_path / "generated.mp4"),
        revision=None,
        force_download=False,
        gen_kwargs={},
        verbose=False,
    )

    with (
        patch.object(
            video_module, "load_video_generation_model", return_value=SimpleNamespace()
        ) as mock_load,
        patch.object(video_module, "generate_video", return_value=_result()),
    ):
        video_module.run_video_generation_cli(args)

    assert mock_load.call_args.kwargs["workflow"] == expected_workflow


def test_generate_video_string_request_routes_extra_kwargs():
    requests = []
    progress_events = []
    model = SimpleNamespace(
        generate=lambda request: (requests.append(request), _result())[1]
    )

    result = video_module.generate_video(
        model,
        "synthetic",
        seed=7,
        num_frames=2,
        progress_callback=lambda *event: progress_events.append(event),
        custom_option="value",
    )

    assert result.seed == 7
    assert requests[0].num_frames == 2
    assert requests[0].extra == {"custom_option": "value"}
    assert progress_events == [("complete", 30, 30, 2)]


def test_video_progress_bar_reports_effective_frames_per_second(monkeypatch):
    class FakeBar:
        def __init__(self):
            self.total = 2
            self.n = 0
            self.descriptions = []
            self.postfix = None
            self.closed = False

        def set_description_str(self, description):
            self.descriptions.append(description)

        def update(self, amount):
            self.n += amount

        def set_postfix(self, postfix, *, refresh):
            self.postfix = postfix

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    bar = FakeBar()
    clock = iter([10.0, 10.0, 11.0, 12.0, 14.0])
    monkeypatch.setattr(video_module, "tqdm", lambda **kwargs: bar)
    monkeypatch.setattr(video_module.time, "perf_counter", lambda: next(clock))

    with video_module._VideoProgressBar(
        steps=2, num_frames=4, disable=False
    ) as progress:
        progress("prepare", 0, 2, 4)
        progress("cache_adaln", 1, 2, 4)
        progress("denoise", 1, 2, 4)
        progress("complete", 2, 2, 4)

    assert bar.n == 2
    assert bar.postfix == {"frames/s": "1.00"}
    assert "Caching AdaLN" in bar.descriptions
    assert "Denoising video" in bar.descriptions
    assert bar.closed


def test_generate_cli_routes_video_before_vlm_load():
    args = Namespace(output_modality="video")

    with (
        patch.object(dispatch_module, "parse_arguments", return_value=args),
        patch.object(dispatch_module, "run_video_generation_cli") as mock_run_video,
        patch.object(dispatch_module, "load") as mock_load,
    ):
        dispatch_module.main()

    mock_run_video.assert_called_once_with(args)
    mock_load.assert_not_called()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the container smoke test",
)
def test_video_result_muxes_video_and_audio(tmp_path):
    frame_count = 6
    fps = 6
    sampling_rate = 32000
    frames = np.zeros((frame_count, 32, 64, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(frame_count, dtype=np.uint8)[:, None, None] * 30
    timeline = np.arange(sampling_rate, dtype=np.float32) / sampling_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * timeline)
    audio = np.stack([tone, tone])
    result = VideoGenerationResult(
        frames=mx.array(frames),
        audio=mx.array(audio),
        fps=fps,
        sampling_rate=sampling_rate,
        seed=7,
        width=64,
        height=32,
        num_frames=frame_count,
        steps=2,
        model="synthetic",
        family="synthetic",
        workflow="t2va",
    )

    output_path = result.save(tmp_path / "muxed.mp4")
    probe = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {
        stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]
    }
    assert output_path.stat().st_size > 0
    assert stream_types == {"video", "audio"}
