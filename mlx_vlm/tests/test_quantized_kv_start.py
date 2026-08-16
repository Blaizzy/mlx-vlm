import pytest

from mlx_vlm.generate.common import (
    DEFAULT_QUANTIZED_KV_START,
    resolve_quantized_kv_start,
)

DEFAULT = DEFAULT_QUANTIZED_KV_START


class TestQuantizedKvStartDefault:
    """Quantizing mid-prefill costs prefill throughput and peak memory.

    The saving only lands at decode, so the default threshold should not fire
    while the prompt is still being ingested.
    """

    def test_long_prompt_defers_past_the_prompt(self):
        assert resolve_quantized_kv_start(4, DEFAULT, DEFAULT * 4) == DEFAULT * 4

    def test_short_prompt_keeps_the_default(self):
        # Below the default nothing changes: the cache is too small for
        # quantization to be worth anything either way.
        assert resolve_quantized_kv_start(4, DEFAULT, DEFAULT // 10) == DEFAULT

    def test_prompt_exactly_at_the_default(self):
        assert resolve_quantized_kv_start(4, DEFAULT, DEFAULT) == DEFAULT

    @pytest.mark.parametrize("explicit", [0, 128, DEFAULT + 1])
    def test_explicit_threshold_is_respected(self, explicit):
        # A caller who names a threshold means it, including one that fires
        # during prefill, and including zero.
        assert resolve_quantized_kv_start(4, explicit, 99999) == explicit

    def test_unquantized_runs_are_untouched(self):
        # With kv_bits=None the threshold is inert; leave it exactly as passed
        # so nothing downstream sees a surprising value.
        assert resolve_quantized_kv_start(None, DEFAULT, 99999) == DEFAULT

    def test_unknown_prompt_length_is_untouched(self):
        assert resolve_quantized_kv_start(4, DEFAULT, None) == DEFAULT

    @pytest.mark.parametrize("bits", [2, 3, 4, 8, 3.5])
    def test_applies_to_every_bit_width(self, bits):
        assert resolve_quantized_kv_start(bits, DEFAULT, DEFAULT * 2) == DEFAULT * 2

    def test_generate_step_uses_the_helper(self):
        # Guard the wiring, not just the arithmetic: the resolved value has to
        # reach the functools.partial that captures it.
        import inspect

        from mlx_vlm.generate import ar as ar_module

        source = inspect.getsource(ar_module.generate_step)
        resolve_at = source.find("resolve_quantized_kv_start(")
        bind_at = source.find("quantize_cache_fn = functools.partial(")
        assert resolve_at != -1, "generate_step no longer resolves the threshold"
        assert bind_at != -1
        assert resolve_at < bind_at, (
            "the threshold must be resolved before quantize_cache_fn binds it, "
            "or the adjustment is silently discarded"
        )
