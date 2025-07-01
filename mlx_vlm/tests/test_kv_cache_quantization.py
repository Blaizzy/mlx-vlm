from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.generate import generate_step


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

    def __call__(self, *args, **kwargs):
        return self.return_value


class MockCache:
    """Mock cache for testing quantization."""

    def __init__(self):
        self.keys = [mx.random.normal((1, 8, 100, 64)) for _ in range(2)]
        self.values = [mx.random.normal((1, 8, 100, 64)) for _ in range(2)]
        self.offset = 0
        self.quantized = False
        self.quantization_params = {}

    def update(self, keys, values):
        """Update cache with new keys and values."""
        self.keys = keys
        self.values = values

    def reset(self):
        """Reset the cache."""
        self.offset = 0

    def __getitem__(self, layer_idx):
        """Get cache for a specific layer."""
        return self


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    return MockCache()


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    processor = Mock()
    processor.tokenizer = Mock()
    processor.tokenizer.eos_token_id = 2
    processor.image_processor = Mock()
    processor.image_processor.size = {"height": 336, "width": 336}
    processor.image_processor.image_mean = [0.48145466, 0.4578275, 0.40821073]
    processor.image_processor.image_std = [0.26862954, 0.26130258, 0.27577711]
    return processor


class TestKVCacheQuantization:
    """Test suite for KV cache quantization functionality."""

    def test_quantize_cache_fn_creation_and_called_during_generation(self):
        """Test that quantize_cache_fn is created correctly and called at the right points during generation."""
        with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
            input_ids = mx.array([[1, 2, 3, 4, 5]])
            pixel_values = mx.random.normal((1, 3, 336, 336))
            mask = mx.ones((1, 5))
            model = MockModel()

            # Generate multiple tokens to test both creation and calling
            gen = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                kv_bits=4,
                kv_group_size=64,
                quantized_kv_start=100,
                max_tokens=3,
            )

            # Consume the generator
            tokens = []
            for token, _ in gen:
                tokens.append(token)

            # Verify that maybe_quantize_kv_cache was called with correct parameters
            assert mock_quantize.called
            call_args = mock_quantize.call_args_list[0]
            assert "quantized_kv_start" in call_args[1]
            assert "kv_group_size" in call_args[1]
            assert "kv_bits" in call_args[1]
            assert call_args[1]["quantized_kv_start"] == 100
            assert call_args[1]["kv_group_size"] == 64
            assert call_args[1]["kv_bits"] == 4

            # Should be called once after initial forward pass and once per generated token
            expected_calls = 1 + len(tokens)
            assert mock_quantize.call_count == expected_calls

    def test_different_quantization_bit_configurations(self):
        """Test KV cache quantization with different bit configurations."""
        bit_configs = [2, 4, 8]

        for bits in bit_configs:
            with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
                input_ids = mx.array([[1, 2, 3, 4, 5]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 5))
                model = MockModel()

                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=bits,
                    max_tokens=1,
                )

                # Consume generator
                try:
                    next(gen)
                except StopIteration:
                    pass

                # Verify quantization was called with correct bit configuration
                call_args = mock_quantize.call_args_list[0]
                assert call_args[1]["kv_bits"] == bits

    def test_quantization_with_different_group_sizes(self):
        """Test KV cache quantization with different group sizes."""
        group_sizes = [64, 128]

        for group_size in group_sizes:
            with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
                input_ids = mx.array([[1, 2, 3, 4, 5]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 5))
                model = MockModel()

                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=4,
                    kv_group_size=group_size,
                    max_tokens=1,
                )

                # Consume generator
                try:
                    next(gen)
                except StopIteration:
                    pass

                # Verify quantization was called with correct group size
                call_args = mock_quantize.call_args_list[0]
                assert call_args[1]["kv_group_size"] == group_size

    def test_quantization_start_index(self):
        """Test that quantization respects the start index parameter."""
        start_indices = [0, 100, 1000, 5000]

        for start_idx in start_indices:
            with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
                input_ids = mx.array([[1, 2, 3, 4, 5]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 5))
                model = MockModel()

                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=4,
                    quantized_kv_start=start_idx,
                    max_tokens=1,
                )

                # Consume generator
                try:
                    next(gen)
                except StopIteration:
                    pass

                # Verify quantization was called with correct start index
                call_args = mock_quantize.call_args_list[0]
                assert call_args[1]["quantized_kv_start"] == start_idx

    def test_generation_without_quantization(self):
        """Test that generation works without KV cache quantization."""
        with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
            input_ids = mx.array([[1, 2, 3, 4, 5]])
            pixel_values = mx.random.normal((1, 3, 336, 336))
            mask = mx.ones((1, 5))
            model = MockModel()

            # Generate without specifying kv_bits (quantization disabled)
            gen = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                max_tokens=2,
            )

            # Consume generator
            tokens = []
            for token, _ in gen:
                tokens.append(token)

            # Quantization should still be called but with kv_bits=None
            assert mock_quantize.call_count > 0
            call_args = mock_quantize.call_args_list[0]
            assert call_args[1]["kv_bits"] is None

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

    def test_quantization_with_long_sequences(self):
        """Test KV cache quantization behavior with long sequences."""
        with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
            # Create a longer input sequence
            input_ids = mx.array([[1] * 1000])  # 1000 tokens
            pixel_values = mx.random.normal((1, 3, 336, 336))
            mask = mx.ones((1, 1000))
            model = MockModel()

            gen = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                kv_bits=4,
                quantized_kv_start=500,  # Start quantization after 500 tokens
                max_tokens=5,
            )

            # Consume generator
            tokens = []
            for token, _ in gen:
                tokens.append(token)

            # Verify quantization was called multiple times
            assert mock_quantize.call_count > 0

    def test_quantization_with_prompt_cache(self):
        """Test KV cache quantization when using existing prompt cache."""
        with patch("mlx_lm.generate.maybe_quantize_kv_cache") as mock_quantize:
            input_ids = mx.array([[1, 2, 3, 4, 5]])
            pixel_values = mx.random.normal((1, 3, 336, 336))
            mask = mx.ones((1, 5))
            model = MockModel()

            # Create a mock prompt cache
            prompt_cache = MockCache()

            gen = generate_step(
                input_ids=input_ids,
                model=model,
                pixel_values=pixel_values,
                mask=mask,
                kv_bits=8,
                prompt_cache=prompt_cache,
                max_tokens=2,
            )

            # Consume generator
            tokens = []
            for token, _ in gen:
                tokens.append(token)

            # Verify quantization was called on the provided cache
            assert mock_quantize.call_count > 0
            # The cache passed to quantization should be the same as provided
            for call in mock_quantize.call_args_list:
                assert call[0][0] is prompt_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
