import argparse
from unittest.mock import patch

from mlx_vlm.models.rfdetr.generate import _get_annotator, main


def _chain_names(annotator):
    return [type(a).__name__ for a in getattr(annotator, "annotators", [annotator])]


def test_segment_chain_drops_boxes_and_labels_when_show_boxes_is_off():
    assert _chain_names(_get_annotator(None, "segment", show_boxes=True)) == [
        "MaskAnnotator",
        "BoxAnnotator",
        "LabelAnnotator",
    ]
    assert _chain_names(_get_annotator(None, "segment", show_boxes=False)) == [
        "MaskAnnotator"
    ]


def test_show_boxes_defaults_to_on():
    assert _chain_names(_get_annotator(None, "segment")) == [
        "MaskAnnotator",
        "BoxAnnotator",
        "LabelAnnotator",
    ]


def test_detect_keeps_boxes_because_nothing_else_is_drawn():
    # A detect chain has nothing but boxes and labels, so turning them off
    # would render an empty overlay.
    for show_boxes in (True, False):
        assert _chain_names(_get_annotator(None, "detect", show_boxes=show_boxes)) == [
            "BoxAnnotator",
            "LabelAnnotator",
        ]


def _capture_parser():
    """Return the ArgumentParser that main() builds."""

    class _Stop(Exception):
        pass

    captured = {}

    def _fake_parse(self, *args, **kwargs):
        captured["parser"] = self
        raise _Stop()

    with patch.object(argparse.ArgumentParser, "parse_args", _fake_parse):
        try:
            main()
        except _Stop:
            pass
    return captured["parser"]


def test_cli_can_turn_boxes_off():
    parser = _capture_parser()

    args = parser.parse_args(["--model", "m", "--image", "i.jpg"])
    assert args.show_boxes is True

    args = parser.parse_args(["--model", "m", "--image", "i.jpg", "--no-show-boxes"])
    assert args.show_boxes is False
