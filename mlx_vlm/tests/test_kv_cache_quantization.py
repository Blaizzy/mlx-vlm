from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

# Import the module to patch
from mlx_vlm.generate import generate_step


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
