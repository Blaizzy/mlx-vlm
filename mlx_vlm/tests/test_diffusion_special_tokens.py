"""Regression tests for issue #1351.

The diffusion detokenization path used to strip *all* ``tokenizer.all_special_ids``
by id, deleting the ``<|tool_call>``/``<|channel>`` markers the server's gemma4
tool parser and thinking splitter anchor on. The fix narrows the skip set to
preserve those structural markers while still dropping pad/mask/EOS sentinels.
"""

import unittest

from mlx_vlm.server.generation import (
    _diffusion_skip_special_token_ids,
    _structural_marker_strings,
)
from mlx_vlm.server.responses_state import _split_thinking, process_tool_calls
from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
from mlx_vlm.tool_parsers import load_tool_module


class _FakeProcessor:
    """Minimal processor whose chat template makes the gemma4 parser get inferred."""

    def __init__(self, chat_template="{{ '<|tool_call>' }}"):
        self.chat_template = chat_template


class _StubTokenizer:
    """Tiny id<->piece tokenizer good enough for skip-set + detok tests."""

    def __init__(self):
        self.vocab_map = {
            0: "<pad>",
            1: "<eos>",
            2: "<unk>",
            10: "<|tool_call>",
            11: "<tool_call|>",
            20: "<|channel>",
            21: "<channel|>",
            30: "thought",
            31: "weighing it",
            32: "call:get_weather{city:Austin}",
            33: "Sunny.",
        }
        # pad/eos/unk plus the four bracket markers are all "special".
        self.all_special_ids = [0, 1, 2, 10, 11, 20, 21]
        self.unk_token_id = 2
        self._tok2id = {v: k for k, v in self.vocab_map.items()}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab_map.get(int(i), "") for i in ids)

    def convert_tokens_to_ids(self, token):
        return self._tok2id.get(token, self.unk_token_id)


class TestDiffusionSkipSpecialTokenIds(unittest.TestCase):
    def test_preserves_markers_but_strips_other_specials(self):
        tok = _StubTokenizer()
        skip = _diffusion_skip_special_token_ids(tok, _FakeProcessor())
        # Tool-call + channel marker ids must NOT be skipped.
        for marker_id in (10, 11, 20, 21):
            self.assertNotIn(marker_id, skip)
        # pad/eos/unk are still stripped.
        for noise_id in (0, 1, 2):
            self.assertIn(noise_id, skip)

    def test_unk_guard(self):
        """A marker absent from the vocab maps to <unk>; <unk> must stay stripped."""
        tok = _StubTokenizer()
        # Drop the channel markers from the vocab so they resolve to <unk>.
        for piece in ("<|channel>", "<channel|>"):
            tok._tok2id.pop(piece, None)
        skip = _diffusion_skip_special_token_ids(tok, _FakeProcessor())
        self.assertIn(tok.unk_token_id, skip)

    def test_no_convert_method_falls_back_to_all_special(self):
        class _NoConvert:
            all_special_ids = [0, 1, 2]

        skip = _diffusion_skip_special_token_ids(_NoConvert(), _FakeProcessor())
        self.assertEqual(skip, {0, 1, 2})

    def test_empty_all_special_ids(self):
        class _Empty:
            all_special_ids = []

        self.assertEqual(
            _diffusion_skip_special_token_ids(_Empty(), _FakeProcessor()), set()
        )

    def test_structural_markers_include_tool_and_channel(self):
        markers = _structural_marker_strings(_FakeProcessor())
        self.assertIn("<|tool_call>", markers)
        self.assertIn("<tool_call|>", markers)
        self.assertIn("<|channel>", markers)
        self.assertIn("<channel|>", markers)


class TestMarkersSurviveDetokenization(unittest.TestCase):
    def _detok(self, tok, token_ids, skip):
        d = NaiveStreamingDetokenizer(tok)
        d.reset()
        for t in token_ids:
            d.add_token(t, skip_special_token_ids=skip)
        d.finalize()
        return d.text

    def test_tool_call_markers_survive_and_parse(self):
        tok = _StubTokenizer()
        skip = _diffusion_skip_special_token_ids(tok, _FakeProcessor())
        # <|tool_call> call:get_weather{...} <tool_call|>  + a stripped <eos>
        text = self._detok(tok, [10, 32, 11, 1], skip)
        self.assertIn("<|tool_call>", text)
        self.assertIn("<tool_call|>", text)
        self.assertNotIn("<eos>", text)

        gemma4 = load_tool_module("gemma4")
        result = process_tool_calls(text, gemma4, tools=None)
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["calls"][0]["function"]["name"], "get_weather")

    def test_channel_markers_survive_and_split(self):
        tok = _StubTokenizer()
        skip = _diffusion_skip_special_token_ids(tok, _FakeProcessor())
        # <|channel> thought <reasoning> <channel|> <content>
        text = self._detok(tok, [20, 30, 31, 21, 33], skip)
        self.assertIn("<|channel>", text)
        reasoning, content = _split_thinking(text)
        self.assertEqual(content, "Sunny.")
        self.assertNotIn("thought", content)

    def test_old_blanket_skip_set_reproduces_the_bug(self):
        """Guard: the previous behaviour (strip all_special_ids) loses the call."""
        tok = _StubTokenizer()
        blanket = set(tok.all_special_ids)
        text = self._detok(tok, [10, 32, 11], blanket)
        self.assertNotIn("<|tool_call>", text)
        gemma4 = load_tool_module("gemma4")
        result = process_tool_calls(text, gemma4, tools=None)
        self.assertEqual(result["calls"], [])  # tool_calls: null, as reported


if __name__ == "__main__":
    unittest.main()
