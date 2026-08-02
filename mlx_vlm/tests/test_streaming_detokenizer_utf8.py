"""Regression test for issue #1771: an incomplete multi-byte UTF-8
sequence at the streaming flush boundary must neither raise nor mangle
the already-decodable head of the buffer."""

import unittest


class FakeTokenizer:
    """Minimal stand-in exposing what BPEStreamingDetokenizer uses."""

    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab(self):
        return self.vocab


class TestBPEStreamingUTF8(unittest.TestCase):
    def test_incomplete_tail_at_flush_boundary(self):
        from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer

        BPEStreamingDetokenizer.make_byte_decoder()
        byte_decoder = BPEStreamingDetokenizer._byte_decoder  # char -> byte
        byte_encoder = {b: c for c, b in byte_decoder.items()}

        def surface(bs: bytes) -> str:
            return "".join(byte_encoder[b] for b in bs)

        # A buffer ending mid-character ("ni" complete + first byte of
        # "hao"), then a space-led token forcing a flush: the old strict
        # decode raised UnicodeDecodeError here; a blanket errors="replace"
        # mangles the head. The fix flushes the clean head and buffers the
        # incomplete tail.
        nihao = "你好".encode("utf-8")                    # 3 + 3 bytes
        vocab = {surface(nihao[:4]): 0, surface(b" ok"): 1}
        detok = BPEStreamingDetokenizer(FakeTokenizer(vocab))
        detok.reset()
        detok.add_token(0)
        detok.add_token(1)   # space-led -> flush with incomplete tail
        detok.finalize()
        self.assertIn("你", detok.text)   # the clean head survived intact
        self.assertIn("ok", detok.text)
        self.assertNotIn("\ufffd", detok.text)  # nothing was mangled


if __name__ == "__main__":
    unittest.main()
