"""CLI wiring for the SAM3 ``--threshold`` flag.

``main()`` dispatches every task with
``args.threshold if args.threshold is not None else <per-task default>``, and
the per-task defaults differ (0.3 for image detect/segment, 0.15 for track,
0.5 for realtime) exactly as the flag's own help text documents.  Those
fallbacks are only reachable while the flag itself defaults to ``None``.
"""

import sys
import unittest
from unittest.mock import patch

from mlx_vlm.models.sam3 import generate as sam3_generate


class TestSam3ThresholdDefaults(unittest.TestCase):
    def _dispatch(self, argv):
        recorded = {}

        def recorder(name):
            def _fn(**kwargs):
                recorded[name] = kwargs

            return _fn

        with (
            patch.object(sam3_generate, "run_image", recorder("run_image")),
            patch.object(sam3_generate, "track_video", recorder("track_video")),
            patch.object(
                sam3_generate, "track_video_realtime", recorder("track_video_realtime")
            ),
            patch.object(sys, "argv", ["generate.py"] + argv),
        ):
            sam3_generate.main()

        self.assertEqual(len(recorded), 1, f"expected one dispatch, got {recorded}")
        return next(iter(recorded.values()))

    def test_image_task_uses_its_documented_default(self):
        kwargs = self._dispatch(
            ["--task", "segment", "--image", "img.png", "--prompt", "a dog"]
        )
        self.assertEqual(kwargs["threshold"], 0.3)

    def test_track_task_uses_its_documented_default(self):
        kwargs = self._dispatch(
            ["--task", "track", "--video", "clip.mp4", "--prompt", "a car"]
        )
        self.assertEqual(kwargs["threshold"], 0.15)

    def test_realtime_task_default_is_unchanged(self):
        kwargs = self._dispatch(["--task", "realtime", "--prompt", "a car"])
        self.assertEqual(kwargs["threshold"], 0.5)

    def test_explicit_threshold_wins_over_the_per_task_default(self):
        kwargs = self._dispatch(
            [
                "--task",
                "track",
                "--video",
                "clip.mp4",
                "--prompt",
                "a car",
                "--threshold",
                "0.9",
            ]
        )
        self.assertEqual(kwargs["threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()
