# Copyright © 2026 Apple Inc.

import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.kernels import lfm3_rope, short_conv_sequence, vision_rotate
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.lfm2 import language as lfm2_language
from mlx_vlm.models.lfm2.language import ShortConv
from mlx_vlm.models.lfm3 import language as lfm3_language
from mlx_vlm.models.lfm3.config import ModelConfig
from mlx_vlm.models.lfm3.language import LanguageModel, RotaryEmbedding

DTYPES = (mx.bfloat16, mx.float16, mx.float32)


class _Args:
    hidden_size = 48
    conv_L_cache = 7
    conv_bias = False


class _Args3(_Args):
    # The LFM2 and LFM2.5 kernel width.
    conv_L_cache = 3


class _ArgsBias(_Args):
    conv_bias = True


def _counting(module, name):
    return mock.patch.object(module, name, wraps=getattr(module, name))


class TestShortConv(unittest.TestCase):
    def _check(self, batch, dtype, args=_Args):
        mx.random.seed(0)
        conv = ShortConv(args(), 0)
        conv.set_dtype(dtype)
        conv.eval()
        fused, ref = ArraysCache(size=1), ArraysCache(size=1)
        with (
            _counting(lfm2_language, "short_conv_sequence") as seq,
            _counting(lfm2_language, "_short_conv_step_kernel") as step,
        ):
            # A first chunk without state, one shorter than the state, then steps.
            for length in (9, 3, 1, 1, 17, 1):
                x = mx.random.normal((batch, length, _Args.hidden_size)).astype(dtype)
                y = conv(x, None, fused)
                y_ref = conv.out_proj(
                    conv._convolve_projected(conv.in_proj(x), None, ref, None)
                )
                self.assertTrue(mx.array_equal(y, y_ref).item(), (dtype, length))
                self.assertTrue(
                    mx.array_equal(fused[0], ref[0]).item(), (dtype, length)
                )
        self.assertEqual(seq.call_count, 3)
        self.assertEqual(step.call_count, 3)

    def test_matches_standard_ops_across_chunks_and_steps(self):
        for dtype in DTYPES:
            self._check(1, dtype)

    def test_batch_of_two(self):
        self._check(2, mx.bfloat16)

    def test_lfm2_kernel_width(self):
        for dtype in DTYPES:
            self._check(1, dtype, _Args3)

    def test_with_bias(self):
        for dtype in DTYPES:
            self._check(1, dtype, _ArgsBias)

    def test_without_state_is_a_zero_padded_conv(self):
        mx.random.seed(1)
        bcx = mx.random.normal((2, 5, 3 * 16)).astype(mx.bfloat16)
        weight = mx.random.normal((16, 4, 1)).astype(mx.bfloat16)
        y, state = short_conv_sequence(bcx, None, weight)
        zeros = mx.zeros((2, 3, 16), mx.bfloat16)
        y0, state0 = short_conv_sequence(bcx, zeros, weight)
        self.assertTrue(mx.array_equal(y, y0).item())
        self.assertTrue(mx.array_equal(state, state0).item())


class TestRotations(unittest.TestCase):
    def test_vision_rotate_matches_reference(self):
        mx.random.seed(2)
        n, h, d = 37, 3, 64
        x = mx.random.normal((n, h, d)).astype(mx.bfloat16)
        angles = 40 * mx.random.uniform(shape=(n, d))
        cos, sin = mx.cos(angles), mx.sin(angles)
        t = x.astype(mx.float32)
        a, b = mx.split(t, 2, axis=-1)
        rotated = t * cos[:, None] + mx.concatenate([-b, a], axis=-1) * sin[:, None]
        ref = rotated.astype(x.dtype).transpose(1, 0, 2)[None]
        self.assertTrue(mx.array_equal(vision_rotate(x, cos, sin), ref).item())

    def test_lfm3_rope_matches_module(self):
        partial = {"rope_theta": 1e6, "partial_rotary_factor": 0.5}
        for params in (partial, {"rope_theta": 1e4}):
            for dtype in (mx.bfloat16, mx.float16):
                for offset, batch in ((0, 1), (4093, 1), (mx.array(77), 1), (5, 2)):
                    mx.random.seed(3)
                    rope = RotaryEmbedding(128, params)
                    x = mx.random.normal((batch, 9, 4, 128)).astype(dtype)
                    ref = rope(x.transpose(0, 2, 1, 3), offset)
                    y = lfm3_rope(x, rope.padded_inv_freq(), offset)
                    self.assertTrue(
                        mx.array_equal(y, ref).item(), (params, dtype, offset)
                    )


class TestLanguageModel(unittest.TestCase):
    def test_fused_paths_are_bitwise_through_chunked_prefill_and_decode(self):
        config = ModelConfig(
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=101,
            layer_types=["conv", "full_attention", "sliding_attention"],
            sliding_window=4,
        )
        mx.random.seed(4)
        model = LanguageModel(config)
        model.set_dtype(mx.bfloat16)
        tokens = mx.random.randint(0, config.vocab_size, (1, 23))

        def run():
            cache = model.make_cache()
            outs = []
            for chunk in (tokens[:, :10], tokens[:, 10:22], tokens[:, 22:]):
                outs.append(model(chunk, cache=cache).logits)
            for step in range(3):
                outs.append(model(tokens[:, step : step + 1], cache=cache).logits)
            mx.eval(outs)
            return outs

        model.eval()
        with (
            _counting(lfm3_language, "lfm3_rope") as rope,
            _counting(lfm3_language.mx.fast, "scaled_dot_product_attention") as sdpa,
        ):
            fused = run()
        self.assertGreater(rope.call_count, 0)
        # Chunks of 32 tokens or fewer keep the band mask.
        windowed = [c for c in sdpa.call_args_list if c.kwargs.get("window_size")]
        self.assertEqual(len(windowed), 0)

        # An MLX without window_size keeps the band mask.
        with (
            mock.patch.object(lfm3_language, "_SDPA_HAS_WINDOW", False),
            _counting(lfm3_language.mx.fast, "scaled_dot_product_attention") as sdpa,
        ):
            without_window = run()
        self.assertFalse(any(c.kwargs.get("window_size") for c in sdpa.call_args_list))
        for a, b in zip(fused, without_window):
            self.assertTrue(mx.array_equal(a, b).item())

        # Training mode takes none of the kernels, which have no gradient.
        model.train()
        with (
            _counting(lfm3_language, "lfm3_rope") as rope,
            _counting(lfm2_language, "short_conv_sequence") as seq,
            _counting(lfm3_language.mx.fast, "scaled_dot_product_attention") as sdpa,
        ):
            reference = run()
        self.assertEqual(rope.call_count, 0)
        self.assertEqual(seq.call_count, 0)
        self.assertFalse(any(c.kwargs.get("window_size") for c in sdpa.call_args_list))
        for a, b in zip(fused, reference):
            self.assertTrue(mx.array_equal(a, b).item())

    def test_window_path_matches_band_mask(self):
        # head_dim 128 reaches MLX's fused attention kernels.
        config = ModelConfig(
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=101,
            layer_types=["sliding_attention"],
            sliding_window=256,
        )
        mx.random.seed(6)
        model = LanguageModel(config)
        model.set_dtype(mx.bfloat16)
        model.eval()
        tokens = mx.random.randint(0, config.vocab_size, (1, 452))

        def run(chunks):
            cache = model.make_cache()
            outs = [model(tokens[:, a:b], cache=cache).logits for a, b in chunks]
            mx.eval(outs)
            return outs

        scenarios = (
            # A first chunk past the window, then later chunks on 255 cached keys;
            # the 5- and 7-token chunks keep the mask.
            (((0, 300), (300, 305), (305, 345), (345, 445), (445, 452)), 3),
            # A first chunk inside the window, then chunks that cross it with
            # 100 and then 255 cached keys.
            (((0, 100), (100, 300), (300, 345)), 2),
        )
        for chunks, expected in scenarios:
            with _counting(
                lfm3_language.mx.fast, "scaled_dot_product_attention"
            ) as sdpa:
                fused = run(chunks)
            windowed = [c for c in sdpa.call_args_list if c.kwargs.get("window_size")]
            if lfm3_language._SDPA_HAS_WINDOW:
                self.assertEqual(len(windowed), expected)
            with mock.patch.object(
                lfm3_language.Attention, "_fused_window_applies", return_value=False
            ):
                reference = run(chunks)
            for a, b in zip(fused, reference):
                self.assertTrue(mx.array_equal(a, b).item())

    def test_backward_in_training_mode(self):
        config = ModelConfig(
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=101,
            layer_types=["conv", "full_attention", "sliding_attention"],
            sliding_window=4,
        )
        mx.random.seed(5)
        model = LanguageModel(config)
        # bfloat16, so the inference kernels would apply if training did not exclude them.
        model.set_dtype(mx.bfloat16)
        model.train()
        tokens = mx.random.randint(0, config.vocab_size, (1, 12))

        def loss(model):
            logits = model(tokens[:, :-1]).logits
            return nn.losses.cross_entropy(logits, tokens[:, 1:]).mean()

        value, grads = nn.value_and_grad(model, loss)(model)
        mx.eval(value, grads)
        self.assertTrue(mx.isfinite(value).item())


if __name__ == "__main__":
    unittest.main()
