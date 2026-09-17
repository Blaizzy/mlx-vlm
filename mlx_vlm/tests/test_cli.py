"""CLI parsing, modality routing, detector options, and terminal display contracts."""

import argparse
import importlib
import inspect
import io
import os
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, call, patch

import mlx.core as mx
import pytest

from mlx_vlm import chat
from mlx_vlm.generate import diffusion, dispatch
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.audio import AudioGenerationResult
from mlx_vlm.generate.common import GenerationResult
from mlx_vlm.models.diffusion_gemma import visualizer
from mlx_vlm.models.diffusion_visualizer import _CanvasRedrawer
from mlx_vlm.models.rfdetr import generate as rfdetr
from mlx_vlm.models.sam3 import generate as sam3
from mlx_vlm.tests.test_diffusion_models import FakeProcessor, make_diffusion_model
from mlx_vlm.tests.test_video_generation import _result as _video_result

image_generation = importlib.import_module("mlx_vlm.generate.image")
video_generation = importlib.import_module("mlx_vlm.generate.video_generation")


def _capture_parser(entrypoint):
    with patch.object(
        argparse.ArgumentParser, "parse_args", autospec=True, side_effect=StopIteration
    ) as parse:
        with pytest.raises(StopIteration):
            entrypoint()
    return parse.call_args.args[0]


def _args(*flags, **overrides):
    with patch.object(sys, "argv", ["mlx_vlm.generate", *flags]):
        return NS(**(vars(dispatch.parse_arguments()) | {"model": "demo"} | overrides))


@contextmanager
def _text_cli(args, processor=None, family="demo"):
    values = dict(
        parse_arguments=args,
        load=(NS(config=NS(model_type=family)), processor or NS()),
        apply_chat_template="prompt",
        generate=NS(text="done"),
        generate_audio=AudioGenerationResult(text="Hello.", path=args.output),
    )
    mocks = {name: Mock(return_value=value) for name, value in values.items()}
    with patch.multiple(dispatch, **mocks):
        yield NS(**mocks)


def _assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected


def test_package_clears_main_thread_mlx_streams_at_exit(tmp_path):
    marker = tmp_path / "streams-cleared"
    script = """
import os
from pathlib import Path
import mlx.core as mx

original_clear_streams = getattr(mx, "clear_streams", None)
def record_clear_streams():
    Path(os.environ["MLX_VLM_CLEAR_STREAMS_MARKER"]).write_text("cleared")
    if original_clear_streams is not None:
        original_clear_streams()

mx.clear_streams = record_clear_streams
import mlx_vlm
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "MLX_VLM_CLEAR_STREAMS_MARKER": str(marker)},
    )
    assert marker.read_text() == "cleared"


@pytest.mark.parametrize(
    "entrypoint,default", [(dispatch.parse_arguments, False), (chat.main, True)]
)
def test_cli_boolean_and_thinking_flags(entrypoint, default):
    parser = _capture_parser(entrypoint)
    assert parser.parse_args([]).verbose is default
    assert parser.parse_args([]).thinking_mode is None
    for flag, expected in [("--verbose", True), ("--no-verbose", False)]:
        assert parser.parse_args([flag]).verbose is expected
    for mode in ("enabled", "disabled", "adaptive"):
        assert parser.parse_args(["--thinking-mode", mode]).thinking_mode == mode
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--thinking-mode", "invalid"])
    assert error.value.code == 2


def test_cli_and_library_agree_on_shared_defaults():
    cli = vars(_args())
    library = {
        name: p.default
        for name, p in inspect.signature(generate_step).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    shared = cli.keys() & library.keys()
    assert len(shared) >= 10  # A rename must not turn this into an empty comparison.
    assert {name: cli[name] for name in shared if cli[name] is not None} == {
        name: library[name] for name in shared if cli[name] is not None
    }


@pytest.mark.parametrize(
    "task,media,target,threshold",
    [
        ("segment", ["--image", "img.png"], "run_image", 0.3),
        ("track", ["--video", "clip.mp4"], "track_video", 0.15),
        ("realtime", [], "track_video_realtime", 0.5),
    ],
)
def test_sam3_task_defaults(task, media, target, threshold):
    calls = {
        name: Mock() for name in ("run_image", "track_video", "track_video_realtime")
    }
    with (
        patch.multiple(sam3, **calls),
        patch.object(
            sys, "argv", ["generate", "--task", task, *media, "--prompt", "a dog"]
        ),
    ):
        sam3.main()
    assert {name: mock.call_count for name, mock in calls.items()} == {
        name: int(name == target) for name in calls
    }
    assert calls[target].call_args.kwargs["threshold"] == threshold


@pytest.mark.parametrize(
    "task,boxes,names",
    [
        ("segment", True, ["MaskAnnotator", "BoxAnnotator", "LabelAnnotator"]),
        ("segment", False, ["MaskAnnotator"]),
        ("detect", True, ["BoxAnnotator", "LabelAnnotator"]),
        ("detect", False, ["BoxAnnotator", "LabelAnnotator"]),
    ],
)
def test_detector_annotations(task, boxes, names):
    parser = _capture_parser(rfdetr.main)
    flags = ["--model", "m", "--image", "i.jpg"]
    assert (
        parser.parse_args(flags + ([] if boxes else ["--no-show-boxes"])).show_boxes
        is boxes
    )
    annotator = rfdetr._get_annotator(None, task, show_boxes=boxes)
    assert [
        type(a).__name__ for a in getattr(annotator, "annotators", [annotator])
    ] == names


def test_diffusion_verbose_statistics(capsys):
    dispatch.generate(
        make_diffusion_model(),
        FakeProcessor(),
        "",
        input_ids=mx.array([[2, 3]], dtype=mx.int32),
        max_tokens=2,
        max_denoising_steps=1,
        verbose=True,
    )
    output = capsys.readouterr().out
    assert all(label in output for label in ("Prompt:", "Generation:", "Peak memory:"))
    assert all(
        label not in output
        for label in ("Diffusion:", "work tokens", "work-tokens-per-sec")
    )


@pytest.mark.parametrize(
    "text,options", [("[Mask]\nHello", {"requested_width": 80}), ("A" * 200, {})]
)
def test_diffusion_display_preserves_text(text, options):
    live = diffusion._format_diffusion_live_text
    draft = GenerationResult(
        is_draft=True,
        draft_text=text,
        diffusion_canvas_index=1,
        diffusion_step=1,
        diffusion_total_steps=4,
    )
    assert diffusion._format_diffusion_draft_line(draft, **options) == text
    assert live(text, **options) == text
    assert live("hello\nworld", 80) == "hello\nworld"
    assert live("hello\nworld", 80, preserve_newlines=False) == "hello\\nworld"
    assert live("A" * 200, 20).endswith("...")


@pytest.mark.parametrize(
    "text,width,expected",
    [
        ("alpha beta gamma delta", 11, "alpha beta\ngamma delta"),
        ("alpha\nbeta gamma", 20, "alpha\nbeta gamma"),
        ("abcdef", 3, "abc\ndef"),
    ],
)
def test_diffusion_word_wrap(text, width, expected):
    assert visualizer._wrap_text(text, width) == expected


def test_diffusion_visualizer_and_handler():
    view = visualizer.DiffusionGemma4Visualizer()
    view.redrawer = Mock()
    view.handle_text("Hello.\n")
    view.handle_draft(NS(draft_text="[Mask]\nworld"))
    view.redrawer.draw.assert_called_with("Hello.\n[Mask]\nworld", wrap_width=None)
    view.handle_text(" Bye.")
    view.redrawer.draw.assert_called_with("Hello.\n Bye.", wrap_width=None)
    view.finish("Hello.\n Bye.")
    view.redrawer.finish.assert_called_once_with()
    options = {}
    model = NS(make_unmasking_visualizer=visualizer.make_unmasking_visualizer)
    with patch("sys.stdout.isatty", return_value=True):
        handler = diffusion.DiffusionOutputHandler(model, options, verbose=True)
    assert handler._model_visualizer is not None and options["diffusion_show_unmasking"]
    assert handler.redrawer is None
    handler._model_visualizer = recorder = Mock()
    recorder.handle_text.return_value = True
    draft = NS(draft_text="[Mask]")
    handler.handle_draft(draft)
    assert handler.handle_text("hi") is True
    handler.finish("hi")
    assert recorder.mock_calls == [
        call.handle_draft(draft),
        call.handle_text("hi"),
        call.finish("hi"),
    ]


@pytest.mark.parametrize(
    "sampler,threshold,expected",
    [
        ("confidence-threshold", None, {}),
        ("entropy-bound", None, {"diffusion_sampler": "entropy-bound"}),
        (
            "entropy-bound",
            0.7,
            dict(
                diffusion_sampler="entropy-bound",
                diffusion_threshold=0.7,
                threshold=0.7,
            ),
        ),
    ],
)
def test_diffusion_arguments(sampler, threshold, expected):
    args = _args(diffusion_sampler=sampler, threshold=threshold)
    assert diffusion.diffusion_kwargs_from_args(args, NS(canvas_length=3)) == expected


@pytest.mark.parametrize(
    "alternate", [False, True], ids=["in_place", "alternate_screen"]
)
def test_diffusion_redrawer(alternate):
    redrawer, buffer = _CanvasRedrawer(min_interval=0.0), io.StringIO()
    with (
        patch(
            "shutil.get_terminal_size",
            return_value=NS(columns=40, lines=6 if alternate else 24),
        ),
        redirect_stdout(buffer),
    ):
        redrawer.draw("word " * 60 if alternate else "one two three\nfour")
        first = buffer.getvalue()
        if not alternate:
            redrawer.draw("one two three\nfour five")
            second = buffer.getvalue()[len(first) :]
        redrawer.finish()
    if alternate:
        assert all(
            code in buffer.getvalue()
            for code in ("\033[?1049h", "\033[?25l", "\033[?1049l", "\033[?25h")
        )
        assert not redrawer.alternate_screen
    else:
        assert "\033[2J" not in first + second and "\033[2K" in first
        assert "\033[1A" in second and redrawer.rows == 0


@pytest.mark.parametrize("mode", ["image", "video", "frames"])
def test_text_cli_media_and_arguments(capsys, mode):
    image = mode == "image"
    args = _args(
        image=["image.png"] if image else None,
        video=None if image else ["clip.mp4"],
        fps=2.0 if image else 1.0,
        resize_shape=[224] if image else None,
        prompt=["Describe this input."],
        max_tokens=12 if image else 8,
        temperature=0.7 if image else 0.0,
        revision="main" if image else None,
        prefill_step_size=128 if image else None,
        enable_thinking=image,
        draft_kind="dflash" if image else None,
        video_max_frames=4,
        system="Be concise." if image else None,
    )
    del args.gen_kwargs  # Programmatic callers can omit newer options.
    processor = NS()
    if mode == "video":
        processor.video_processor = NS()
        processor.process = lambda text=None, images=None, videos=None, **kw: None
    frames = [object() for _ in range(6)]
    with (
        patch("mlx_vlm.generate.video.sample_video_frames", return_value=(frames, 2.0)),
        _text_cli(args, processor, "demo" if image else "gemma4") as mocks,
    ):
        dispatch.main()
    assert args.gen_kwargs == {}
    template, generated = mocks.apply_chat_template.call_args, mocks.generate.call_args
    if image:
        assert template.args[2] == [
            {"role": "system", "content": "Be concise."},
            "Describe this input.",
        ]
        for kwargs in (template.kwargs, generated.kwargs):
            assert kwargs["enable_thinking"] is True and "thinking_mode" not in kwargs
        _assert_fields(
            generated.kwargs, max_tokens=12, temperature=0.7, prefill_step_size=128
        )
    elif mode == "video":
        for kwargs in (template.kwargs, generated.kwargs):
            _assert_fields(kwargs, video=["clip.mp4"], fps=1.0)
    else:
        assert template.kwargs["num_images"] == 4 and "video" not in template.kwargs
        assert len(generated.kwargs["image"]) == 4 and generated.kwargs["video"] is None
    output = capsys.readouterr().out
    if mode == "frames":
        assert "no native video support" in output and "4 of 6 sampled frames" in output
    else:
        assert output.strip() == "done"


@pytest.mark.parametrize("modality", ["image", "video"])
def test_modality_routing_precedes_vlm_load(modality):
    args = NS(output_modality=modality)
    with (
        patch.object(dispatch, "parse_arguments", return_value=args),
        patch.object(dispatch, f"run_{modality}_generation_cli") as run,
        patch.object(dispatch, "load") as load,
    ):
        dispatch.main()
    run.assert_called_once_with(args)
    load.assert_not_called()


@pytest.mark.parametrize("task", ["edit", "generate"])
def test_image_cli_request_and_output(tmp_path, task):
    edit = task == "edit"
    output = tmp_path / "image.png"
    extras = (
        {}
        if edit
        else dict(
            gen_kwargs={"sampler_preset": "V4_TURBO_12"},
            prompt_expansion_model="tiny-text-model",
        )
    )
    args = _args(
        model=(
            "black-forest-labs/FLUX.2-klein-9b-kv"
            if edit
            else "ideogram-ai/ideogram-4-fp8"
        ),
        task=task,
        prompt=["add", "sunglasses"] if edit else ["caption"],
        image=["reference.png"] if edit else None,
        output=str(output),
        size="256x512" if edit else "256x256",
        steps=2 if edit else 4,
        seed=7,
        **extras,
    )
    result = NS(path=output, seed=7, width=256, height=512, steps=2, variant="tiny")
    with (
        patch.object(image_generation, "load_image_model", return_value=object()),
        patch.object(
            image_generation, "generate_image", return_value=result
        ) as generate,
    ):
        image_generation.run_image_generation_cli(args)
    request = generate.call_args.args[1]
    if edit:
        assert request.prompt == "add sunglasses"
        assert request.image_paths == ("reference.png",)
        assert (request.width, request.height) == (256, 512)
    else:
        assert request.extra == dict(
            sampler_preset="V4_TURBO_12", prompt_expansion_model="tiny-text-model"
        )
    _assert_fields(generate.call_args.kwargs, task=task, output_path=output)


def test_audio_cli_uses_tts_template_and_shared_loader(tmp_path):
    args = _args(
        "--output-modality",
        "audio",
        "--output",
        str(tmp_path / "out.wav"),
        "--ref-audio",
        "voice.wav",
        "--prompt",
        "Say hello.",
    )
    with _text_cli(args, family="minicpmo") as mocks:
        dispatch.main()
    assert mocks.apply_chat_template.call_args.kwargs["use_tts_template"] is True
    assert mocks.generate_audio.call_args.args[2] == "prompt"
    assert mocks.generate_audio.call_args.kwargs["ref_audio_path"] == "voice.wav"


@pytest.mark.parametrize(
    "flags,message",
    [
        ([], "--output is required"),
        (["--output", "out.wav", "--chat"], "does not support --chat"),
    ],
)
def test_invalid_audio_cli_fails_before_loading(flags, message):
    with _text_cli(_args("--output-modality", "audio", *flags)) as mocks:
        with pytest.raises(ValueError, match=message):
            dispatch.main()
    mocks.load.assert_not_called()


@pytest.mark.parametrize(
    "mode,image,last,workflow",
    [
        ("references", None, None, "ref2va"),
        ("text", None, None, "t2va"),
        ("first", ["first.png"], None, "fl2va"),
        ("last", None, "last.png", "fl2va"),
    ],
)
def test_video_generation_cli(tmp_path, capsys, mode, image, last, workflow):
    references = mode == "references"
    output = tmp_path / "generated.mp4"
    args = _args(
        model="MiniMaxAI/MiniMax-H3",
        prompt=["A", "short", "film"],
        image=image,
        last_image=last,
        reference=(
            ["image=character.png", "video=motion.mp4", "audio=voice.wav"]
            if references
            else None
        ),
        size="64x32" if references else None,
        seed=7,
        steps=2 if references else None,
        num_frames=124 if references else None,
        output=str(output),
        revision="test-revision" if references else None,
        gen_kwargs={"test": True} if references else {},
        verbose=references,
    )
    progress = MagicMock() if references else video_generation._VideoProgressBar
    with (
        patch.object(video_generation, "_VideoProgressBar", progress),
        patch.object(
            video_generation, "load_video_generation_model", return_value=NS()
        ) as load,
        patch.object(
            video_generation, "generate_video", return_value=_video_result(output)
        ) as generate,
    ):
        video_generation.run_video_generation_cli(args)
    assert load.call_args.kwargs["workflow"] == workflow
    if references:
        request = generate.call_args.args[1]
        assert request.prompt == "A short film"
        assert [(item.kind, str(item.path)) for item in request.references] == [
            ("image", "character.png"),
            ("video", "motion.mp4"),
            ("audio", "voice.wav"),
        ]
        assert (request.width, request.height, request.num_frames) == (64, 32, 124)
        callback = progress.return_value.__enter__.return_value
        progress.assert_called_once_with(steps=2, num_frames=124, disable=False)
        callback.assert_any_call("load", 0, 2, 124)
        assert request.progress_callback is callback
        assert generate.call_args.kwargs["output_path"] == output
        stdout = capsys.readouterr().out
        assert "workflow=ref2va" in stdout and "generation_fps=" in stdout
