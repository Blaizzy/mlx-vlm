"""Regression tests for the --verbose CLI flag on chat / generate / video_generate.

Previously these entry points used ``action="store_false"`` for ``--verbose``,
which silently inverted the flag: passing ``--verbose`` actually disabled
verbose output (default was True; the flag flipped it to False). The fix
switches to ``argparse.BooleanOptionalAction`` so that:

* default ``verbose=True`` is preserved (unchanged user-visible default),
* ``--verbose`` keeps verbose on (idempotent, matches the help text),
* ``--no-verbose`` opts out cleanly.

These tests build the parser the same way each ``main()`` does, then assert
the resulting Namespace, so they cover the regression without needing to
load a model.
"""

import argparse
import unittest


def _build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLX Vision Chat CLI")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream tokens as they are generated (use --no-verbose to disable).",
    )
    return parser


def _build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detailed output (use --no-verbose to print only the final result).",
    )
    return parser


def _build_video_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print verbose output (use --no-verbose to print only the final result).",
    )
    return parser


class TestVerboseFlagSemantics(unittest.TestCase):
    """The --verbose flag must not be inverted on any CLI entry point."""

    def _check(self, parser: argparse.ArgumentParser, label: str) -> None:
        with self.subTest(label=label, args=[]):
            self.assertTrue(parser.parse_args([]).verbose)
        with self.subTest(label=label, args=["--verbose"]):
            self.assertTrue(parser.parse_args(["--verbose"]).verbose)
        with self.subTest(label=label, args=["--no-verbose"]):
            self.assertFalse(parser.parse_args(["--no-verbose"]).verbose)

    def test_chat_parser(self) -> None:
        self._check(_build_chat_parser(), "chat")

    def test_generate_parser(self) -> None:
        self._check(_build_generate_parser(), "generate")

    def test_video_generate_parser(self) -> None:
        self._check(_build_video_generate_parser(), "video_generate")


class TestRealParsersMirrorTestParsers(unittest.TestCase):
    """The real entry-point modules must keep --verbose using BooleanOptionalAction.

    We import lazily and skip if mlx isn't available, so this still runs in
    an environment that doesn't have the full GPU stack.
    """

    def _assert_boolean_optional(self, parser: argparse.ArgumentParser) -> None:
        action = next(a for a in parser._actions if "--verbose" in a.option_strings)
        self.assertIsInstance(action, argparse.BooleanOptionalAction)
        self.assertTrue(action.default)

    def test_chat_module_uses_boolean_optional(self) -> None:
        try:
            import inspect

            from mlx_vlm import chat as chat_mod
        except ImportError:
            self.skipTest("mlx_vlm.chat unavailable in this env")
        src = inspect.getsource(chat_mod.main)
        self.assertIn("argparse.BooleanOptionalAction", src)
        self.assertNotIn('"--verbose", action="store_false"', src)

    def test_generate_module_uses_boolean_optional(self) -> None:
        try:
            import inspect

            from mlx_vlm import generate as gen_mod
        except ImportError:
            self.skipTest("mlx_vlm.generate unavailable in this env")
        # generate.py defines parse_arguments() rather than building the parser inline.
        src = inspect.getsource(gen_mod.parse_arguments)
        self.assertIn("argparse.BooleanOptionalAction", src)
        self.assertNotIn('"--verbose", action="store_false"', src)

    def test_video_generate_module_uses_boolean_optional(self) -> None:
        try:
            import inspect

            from mlx_vlm import video_generate as vg_mod
        except ImportError:
            self.skipTest("mlx_vlm.video_generate unavailable in this env")
        src = inspect.getsource(vg_mod.main)
        self.assertIn("argparse.BooleanOptionalAction", src)
        self.assertNotIn('"--verbose", action="store_false"', src)


if __name__ == "__main__":
    unittest.main()
