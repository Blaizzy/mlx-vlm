"""Quantized KV cache lifecycle and batched attention masks."""

from __future__ import annotations

from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate import generate_step
from mlx_vlm.models.base import (
    align_attention_mask_to_scores,
    quantized_scaled_dot_product_attention,
)
from mlx_vlm.models.cache import BatchQuantizedKVCache, create_causal_mask

# Generation with quantized caches

# Import the module to patch


class MockInputEmbeddingsFeatures:
    """Mock input embeddings features object."""

    def __init__(self, inputs_embeds):
        self.inputs_embeds = inputs_embeds
        self.attention_mask_4d = None
        self.cross_attention_states = None
        self.cross_attention_mask = None
        self.full_text_row_masked_out_mask = None
        self.decoder_inputs_embeds = None

    def to_dict(self):
        """Return dictionary of attributes for kwargs expansion."""
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask_4d": self.attention_mask_4d,
            "cross_attention_states": self.cross_attention_states,
            "cross_attention_mask": self.cross_attention_mask,
            "full_text_row_masked_out_mask": self.full_text_row_masked_out_mask,
            "decoder_inputs_embeds": self.decoder_inputs_embeds,
        }


class MockOutput:
    """Mock output object for model calls."""

    def __init__(self):
        self.logits = mx.random.normal((1, 1, 32000))
        self.cross_attention_states = None
        self.encoder_outputs = None


class MockModel(nn.Module):
    """Mock model for testing KV cache quantization."""

    def __init__(self):
        super().__init__()
        # Mock the language_model attribute
        self.language_model = Mock()
        self.language_model.side_effect = self._language_model_call
        self.language_model.layers = [Mock() for _ in range(2)]
        self.language_model.args = Mock()
        self.language_model.args.num_hidden_layers = 2
        self.language_model.head_dim = 64
        self.language_model.n_heads = 8
        self.language_model.n_kv_heads = 8

        # Create a return_value attribute for tests that need to modify it
        self.return_value = MockOutput()

    def _language_model_call(self, *args, **kwargs):
        """Mock call for language_model."""
        return MockOutput()

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        """Mock get_input_embeddings method."""
        # Return mock embeddings with shape (batch, seq_len, hidden_dim)
        batch_size, seq_len = input_ids.shape
        inputs_embeds = mx.random.normal((batch_size, seq_len, 768))
        return MockInputEmbeddingsFeatures(inputs_embeds)

    def __call__(self, *args, **kwargs):
        return self.return_value


class MockCacheLayer:
    """Mock cache layer that supports indexing."""

    def __init__(self):
        self.offset = 0
        self.keys = mx.random.normal((1, 8, 100, 64))
        self.values = mx.random.normal((1, 8, 100, 64))

    @property
    def state(self):
        return self.keys, self.values


class MockCache:
    """Mock cache for testing quantization that supports list-like operations."""

    def __init__(self):
        self.layers = [MockCacheLayer() for _ in range(2)]
        self.offset = 0
        self.quantized = False
        self.quantization_params = {}

    def __getitem__(self, idx):
        """Support indexing to mimic list behavior."""
        return self.layers[idx]

    def __len__(self):
        """Support len() operation."""
        return len(self.layers)

    def update(self, keys, values):
        """Update cache with new keys and values."""
        for i, (k, v) in enumerate(zip(keys, values)):
            if i < len(self.layers):
                self.layers[i].keys = k
                self.layers[i].values = v

    def reset(self):
        """Reset the cache."""
        self.offset = 0
        for layer in self.layers:
            layer.offset = 0


class TestKVCacheQuantization:
    """Test suite for KV cache quantization functionality."""

    def test_cache_memory_with_quantization(self):
        """Test that cache memory usage is affected by quantization."""
        with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
            mock_cache_instance = MockCache()
            mock_make_cache.return_value = mock_cache_instance

            input_ids = mx.array([[1, 2, 3, 4, 5]])
            pixel_values = mx.random.normal((1, 3, 336, 336))
            mask = mx.ones((1, 5))
            model = MockModel()

            # Test with quantization
            gen = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                kv_bits=4,
                max_tokens=1,
            )

            # Consume generator
            try:
                next(gen)
            except StopIteration:
                pass

            # Verify cache was created with max_kv_size if specified
            gen_with_max_kv = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                kv_bits=4,
                max_kv_size=1024,
                max_tokens=1,
            )

            try:
                next(gen_with_max_kv)
            except StopIteration:
                pass

            # Check that make_prompt_cache was called with max_kv_size
            calls = mock_make_cache.call_args_list
            assert any(call[1].get("max_kv_size") == 1024 for call in calls)


class TestQuantizedCacheInHandRolledAttention:
    """Models whose attention cannot go through scaled_dot_product_attention.

    ``QuantizedKVCache.update_and_fetch`` hands back ``(packed, scales,
    biases)`` tuples. Attention paths that build the scores by hand -- gemma2
    because of its logit softcapping, phi3small because of its block-sparse
    mask -- have to materialize that state before touching it, or every call
    dies in ``mx.expand_dims`` on a tuple.
    """

    @pytest.mark.parametrize("kv_bits", [None, 4])
    def test_gemma2_attention_with_quantized_cache(self, kv_bits):
        from mlx_vlm.models.cache import KVCache, QuantizedKVCache
        from mlx_vlm.models.gemma2.config import ModelConfig
        from mlx_vlm.models.gemma2.language import Attention

        args = ModelConfig(
            model_type="gemma2",
            hidden_size=128,
            num_hidden_layers=1,
            intermediate_size=256,
            num_attention_heads=8,
            head_dim=64,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=4,
        )
        attention = Attention(args)
        assert attention.repeats > 1

        cache = KVCache() if kv_bits is None else QuantizedKVCache(bits=kv_bits)
        x = mx.random.normal((1, 6, args.hidden_size))

        out = attention(x, mask=None, cache=cache)

        assert out.shape == (1, 6, args.hidden_size)
        assert not mx.any(mx.isnan(out))

    @pytest.mark.parametrize("kv_bits", [None, 4])
    def test_phi3small_block_sparse_attention_with_quantized_cache(self, kv_bits):
        from mlx_vlm.models.cache import KVCache, QuantizedKVCache
        from mlx_vlm.models.phi3small.config import ModelConfig
        from mlx_vlm.models.phi3small.language import Attention

        args = ModelConfig(
            model_type="phi3small",
            hidden_size=256,
            dense_attention_every_n_layers=2,
            ff_intermediate_size=512,
            gegelu_limit=20.0,
            num_hidden_layers=2,
            num_attention_heads=4,
            layer_norm_epsilon=1e-5,
            vocab_size=32,
            num_key_value_heads=2,
        )
        # layer 0 is a block-sparse layer, which is the hand-rolled path.
        attention = Attention(args, layer_idx=0)
        assert attention.block_sparse

        cache = KVCache() if kv_bits is None else QuantizedKVCache(bits=kv_bits)
        x = mx.random.normal((1, 6, args.hidden_size))

        out = attention(x, mask=None, cache=cache)

        assert out.shape == (1, 6, args.hidden_size)
        assert not mx.any(mx.isnan(out))


# Batch cache state and lifecycle

B, H, D = 2, 4, 64  # batch, heads, head_dim
GROUP_SIZE = 32
BITS = 8


def _rand_kv(batch, seq_len):
    """Return random (keys, values) tensors."""
    k = mx.random.normal((batch, H, seq_len, D))
    v = mx.random.normal((batch, H, seq_len, D))
    return k, v


class TestExtend:
    def test_extend_handles_filtered_non_step_aligned_capacity(self):
        c1 = BatchQuantizedKVCache([7, 7], group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(2, 512)
        c1.update_and_fetch(k1, v1)
        mx.eval(c1.keys)

        # Filtering rows can trim common left padding and leave a backing
        # sequence length that is no longer aligned to the allocation step.
        c1.filter(mx.array([0], mx.int32))
        mx.eval(c1.keys)
        assert c1.keys[0].shape[-2] == 505
        assert c1._idx == 505

        c2 = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k2, v2 = _rand_kv(1, 500)
        c2.update_and_fetch(k2, v2)
        mx.eval(c2.keys)
        assert c2.keys[0].shape[-2] == 512
        assert c2._idx == 500

        c1.extend(c2)
        mx.eval(c1.keys)

        assert c1.keys[0].shape[0] == 2
        assert c1.keys[0].shape[-2] == 512
        assert c1._idx == 505
        assert c1.left_padding.tolist() == [0, 5]


class TestState:
    def test_state_roundtrip(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        state = cache.state
        assert len(state) == 4  # keys, values, offset, left_padding

        cache2 = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        cache2.state = state
        assert cache2._idx == 4

    def test_empty_state(self):
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        state = cache.state
        assert state[0] is None
        assert state[1] is None


class TestPrepareFinalize:
    """Multi-row right-pad lifecycle parity with BatchKVCache (#1567 / #1562)."""

    def test_finalize_noop_without_prepare(self):
        cache = BatchQuantizedKVCache([1, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(B, 4)
        cache.update_and_fetch(k, v)
        before = cache.left_padding.tolist()
        cache.finalize()
        assert cache.left_padding.tolist() == before


# Attention mask alignment

GROUP = 64


class TestAlignAttentionMaskToScores:
    def test_str_passthrough(self):
        scores = mx.zeros((2, 8, 2, 4, 4))
        assert align_attention_mask_to_scores("causal", scores) == "causal"


# Batched attention geometry


def _quant_kv(B, n_kv, L, D, dtype=mx.float16):
    keys = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    values = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    return (
        mx.quantize(keys, group_size=GROUP, bits=BITS),
        mx.quantize(values, group_size=GROUP, bits=BITS),
    )


def _run_sdpa(B, n_q, n_kv, L, K_cache, mask, D=GROUP):
    """K_cache is total key length (offset + L); keys filled to K_cache."""
    queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    q_keys, q_values = _quant_kv(B, n_kv, K_cache, D)
    out = quantized_scaled_dot_product_attention(
        queries, q_keys, q_values, scale=D**-0.5, mask=mask, group_size=GROUP, bits=BITS
    )
    mx.eval(out)
    assert out.shape == (B, n_q, L, D)
    assert mx.isfinite(out).all()
    return out


# ---------------------------------------------------------------------------
# Parametric score/mask geometries
# ---------------------------------------------------------------------------

# (B, n_q, n_kv) layouts seen or plausible in MLX VLMs
HEAD_LAYOUTS = [
    (2, 16, 8),  # Qwen3-0.6B-like GQA (server repro family)
    (2, 32, 8),  # stronger GQA
    (2, 16, 2),  # wider repeat
    (2, 16, 1),  # MQA
    (2, 8, 8),  # MHA n_repeats=1
    (3, 16, 8),  # odd batch
    (4, 16, 8),  # larger batch
    (8, 16, 8),  # B == n_kv (latent mis-align case pre-fix)
    (1, 16, 8),  # single row control
]


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (3, 16, 8), (2, 16, 1)])
def test_left_and_right_padding_together(B, n_q, n_kv):
    L, offset = 16, 0
    left = mx.array([2, 0] + [0] * (B - 2))
    right = mx.array([0, 3] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, left_padding=left[:B], right_padding=right[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("window", [4, 8, 32])
@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (2, 8, 8)])
def test_sliding_window_causal_with_left_pad(window, B, n_q, n_kv):
    L, offset = 24, 40
    pads = mx.array([1, 0] if B == 2 else [1, 0] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, window_size=window, left_padding=pads[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (4, 32, 8)])
def test_additive_float_mask(B, n_q, n_kv):
    L = 12
    causal = create_causal_mask(L, left_padding=mx.array([i % 2 for i in range(B)]))
    mask = mx.where(
        causal, mx.array(0.0, dtype=mx.float16), mx.array(-1e4, dtype=mx.float16)
    )
    _run_sdpa(B, n_q, n_kv, L, L, mask)


# ---------------------------------------------------------------------------
# align helper contract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cache make_mask → quant SDPA (integration of the two layers)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# prepare/finalize stress
# ---------------------------------------------------------------------------


def test_prepare_left_padding_on_empty_only():
    cache = BatchQuantizedKVCache([0, 0], group_size=GROUP, bits=BITS)
    cache.prepare(left_padding=[2, 1])
    assert cache.left_padding.tolist() == [2, 1]
    k = mx.random.normal((2, 2, 4, GROUP))
    v = mx.random.normal((2, 2, 4, GROUP))
    cache.update_and_fetch(k, v)
    with pytest.raises(ValueError, match="empty"):
        cache.prepare(left_padding=[1, 0])


# ---------------------------------------------------------------------------
# Brute force small grid (catch "weird" combos)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
