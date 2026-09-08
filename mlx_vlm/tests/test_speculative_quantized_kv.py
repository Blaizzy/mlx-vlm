"""Speculative decoding must honour --kv-bits, and work on quantized state.

Single-row MTP used to build its prompt cache with
``cache.make_prompt_cache``, bypassing the quantized cache that ``--kv-bits``
asks for: a 32K context then cost ~8.4GB of fp16 KV where 4-bit TurboQuant
needs ~2.1GB. Routing it through ``make_cache`` exposed a second gap --
target verification narrows the cache per draft token, which the quantized
state proxy did not support.
"""

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.muse_glimmer import Model as MuseGlimmerModel
from mlx_vlm.models.muse_glimmer import ModelConfig as MuseGlimmerConfig
from mlx_vlm.models.muse_glimmer import TextConfig, VisionConfig
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    Model as MuseGlimmerAssistantModel,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    ModelConfig as AssistantConfig,
)
from mlx_vlm.speculative.utils import make_speculative_prompt_cache
from mlx_vlm.turboquant import BatchTurboQuantKVCache, _state_length

H, D = 4, 64
BITS = 4
SCALE = D**-0.5


def _filled(seq_len, left_padding=(0,)):
    cache = BatchTurboQuantKVCache(list(left_padding), bits=BITS)
    batch = len(left_padding)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, H, seq_len, D)),
        mx.random.normal((batch, H, seq_len, D)),
    )
    return cache, keys, values


class TestPromptCacheHonoursKvBits:
    """Every speculative configuration must go through ``make_cache``."""

    @pytest.mark.parametrize(
        "draft_kind,batch_size",
        [
            ("mtp", 1),
            ("mtp", 2),
            (None, 1),
            ("eagle3", 1),
            ("dflash", 1),
            ("dflash", 2),
        ],
    )
    def test_uses_supplied_make_cache(self, draft_kind, batch_size):
        sentinel = object()
        calls = []

        def make_cache(lm, left_padding):
            calls.append((lm, left_padding))
            return sentinel

        left_padding = [0] * batch_size
        result = make_speculative_prompt_cache(
            "lm",
            draft_kind=draft_kind,
            batch_size=batch_size,
            left_padding=left_padding,
            make_cache=make_cache,
        )

        # Single-row MTP used to return an unquantized cache built elsewhere.
        assert result is sentinel
        assert calls == [("lm", left_padding)]


def _tiny_dflash_pair():
    """The published Muse Glimmer shapes, scaled down to run in-process."""
    text = TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    target = MuseGlimmerModel(
        MuseGlimmerConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=7,
            video_token_id=6,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    )
    drafter = MuseGlimmerAssistantModel(
        AssistantConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=128,
            sliding_window=8,
            block_size=4,
            mask_token_id=63,
            target_layer_ids=[0, 1],
            num_target_layers=2,
            vocab_size=64,
        )
    )
    return target, drafter


class TestDFlashRunsOnQuantizedKv:
    """DFlash speculation decodes against the quantized cache, not fp16."""

    def test_generate_step_keeps_the_quantized_cache(self):
        target, drafter = _tiny_dflash_pair()
        caches = []

        def make_cache(lm, left_padding):
            built = [
                BatchTurboQuantKVCache(list(left_padding), bits=BITS)
                for _ in lm.language_model.layers
            ]
            caches.append(built)
            return built

        prompt_cache = make_speculative_prompt_cache(
            target,
            draft_kind="dflash",
            batch_size=1,
            left_padding=[0],
            make_cache=make_cache,
        )
        assert prompt_cache is caches[0]
        assert all(isinstance(c, BatchTurboQuantKVCache) for c in prompt_cache)

        tokens = [
            int(token.item()) if hasattr(token, "item") else int(token)
            for token, _ in generate_step(
                mx.array([[1, 2, 3, 4]], dtype=mx.int32),
                target,
                None,
                None,
                max_tokens=6,
                temperature=0,
                prefill_step_size=None,
                draft_model=drafter,
                draft_kind="dflash",
                prompt_cache=prompt_cache,
            )
        ]

        assert len(tokens) == 6
        # Decoding through a drafter must not swap the cache back to fp16.
        assert all(isinstance(c, BatchTurboQuantKVCache) for c in prompt_cache)
        assert prompt_cache[0].bits == BITS


class TestQuantizedStateSlicing:
    """Target verification narrows the cache one draft token at a time."""

    def test_slice_reports_narrowed_length(self):
        _, keys, _ = _filled(300)
        for n in (1, 128, 300):
            narrowed = keys[:, :, :n, :]
            assert narrowed.shape[2] == n
            assert _state_length(narrowed._state) == n

    def test_slice_stays_quantized(self):
        # A slice that dequantized would defeat the point of --kv-bits.
        _, keys, _ = _filled(64)
        assert type(keys[:, :, :32, :]) is type(keys)

    def test_sliced_state_matches_dequantized_prefix(self):
        cache, keys, values = _filled(300)
        mx.eval(cache.keys, cache.values)
        queries = mx.random.normal((1, H, 1, D))

        prefix_keys, prefix_values = keys[:, :, :128, :], values[:, :, :128, :]
        out = scaled_dot_product_attention(
            queries, prefix_keys, prefix_values, cache=cache, scale=SCALE, mask=None
        )
        dq_k, dq_v = cache.dequantize(prefix_keys, prefix_values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=None,
        )
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()

    @pytest.mark.parametrize(
        "key",
        [
            (slice(None), slice(None), slice(5, 10), slice(None)),  # offset start
            (slice(None), 0, slice(None, 10), slice(None)),  # indexes a head
            (slice(None), slice(None), slice(None, 10, 2), slice(None)),  # strided
        ],
    )
    def test_rejects_unsupported_indexing(self, key):
        _, keys, _ = _filled(64)
        with pytest.raises(TypeError):
            keys[key]
