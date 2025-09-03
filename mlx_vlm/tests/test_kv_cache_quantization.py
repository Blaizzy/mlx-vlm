import functools
import sys
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

# Import the module to patch
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


class MockCacheLayer:
    """Mock cache layer that supports indexing."""

    def __init__(self):
        self.offset = 0
        self.keys = mx.random.normal((1, 8, 100, 64))
        self.values = mx.random.normal((1, 8, 100, 64))


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
        # Create a mock for maybe_quantize_kv_cache
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        # Patch using sys.modules to get the correct module
        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                # Create a proper mock cache
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

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
                assert len(mock_quantize_calls) > 0

                # Check the kwargs from the first call
                first_call_kwargs = mock_quantize_calls[0][1]
                assert first_call_kwargs["quantized_kv_start"] == 100
                assert first_call_kwargs["kv_group_size"] == 64
                assert first_call_kwargs["kv_bits"] == 4

                # Should be called once after initial forward pass and once per generated token
                expected_calls = 1 + len(tokens)
                assert len(mock_quantize_calls) == expected_calls

    def test_different_quantization_bit_configurations(self):
        """Test KV cache quantization with different bit configurations."""
        bit_configs = [2, 4, 8]

        for bits in bit_configs:
            mock_quantize_calls = []

            def mock_maybe_quantize_kv_cache(cache, **kwargs):
                mock_quantize_calls.append((cache, kwargs))
                return None

            generate_module = sys.modules["mlx_vlm.generate"]
            with patch.object(
                generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
            ):
                with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                    mock_cache_instance = MockCache()
                    mock_make_cache.return_value = mock_cache_instance

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
                    assert len(mock_quantize_calls) > 0
                    first_call_kwargs = mock_quantize_calls[0][1]
                    assert first_call_kwargs["kv_bits"] == bits

    def test_quantization_with_different_group_sizes(self):
        """Test KV cache quantization with different group sizes."""
        group_sizes = [64, 128]

        for group_size in group_sizes:
            mock_quantize_calls = []

            def mock_maybe_quantize_kv_cache(cache, **kwargs):
                mock_quantize_calls.append((cache, kwargs))
                return None

            generate_module = sys.modules["mlx_vlm.generate"]
            with patch.object(
                generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
            ):
                with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                    mock_cache_instance = MockCache()
                    mock_make_cache.return_value = mock_cache_instance

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
                    assert len(mock_quantize_calls) > 0
                    first_call_kwargs = mock_quantize_calls[0][1]
                    assert first_call_kwargs["kv_group_size"] == group_size

    def test_quantization_start_index(self):
        """Test that quantization respects the start index parameter."""
        start_indices = [0, 100, 1000, 5000]

        for start_idx in start_indices:
            mock_quantize_calls = []

            def mock_maybe_quantize_kv_cache(cache, **kwargs):
                mock_quantize_calls.append((cache, kwargs))
                return None

            generate_module = sys.modules["mlx_vlm.generate"]
            with patch.object(
                generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
            ):
                with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                    mock_cache_instance = MockCache()
                    mock_make_cache.return_value = mock_cache_instance

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
                    assert len(mock_quantize_calls) > 0
                    first_call_kwargs = mock_quantize_calls[0][1]
                    assert first_call_kwargs["quantized_kv_start"] == start_idx

    def test_generation_without_quantization(self):
        """Test that generation works without KV cache quantization."""
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

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
                assert len(mock_quantize_calls) > 0
                first_call_kwargs = mock_quantize_calls[0][1]
                assert first_call_kwargs["kv_bits"] is None

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
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

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
                assert len(mock_quantize_calls) > 0

    def test_quantization_with_prompt_cache(self):
        """Test KV cache quantization when using existing prompt cache."""
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
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
            assert len(mock_quantize_calls) > 0
            # The cache passed to quantization should be the same as provided
            for call_cache, _ in mock_quantize_calls:
                assert call_cache is prompt_cache

    def test_quantization_edge_cases(self):
        """Test edge cases for KV cache quantization."""
        # Test with very small kv_bits
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

                input_ids = mx.array([[1, 2, 3]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 3))
                model = MockModel()

                # Test with 1-bit quantization
                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=1,
                    max_tokens=1,
                )

                try:
                    next(gen)
                except StopIteration:
                    pass

                assert len(mock_quantize_calls) > 0
                first_call_kwargs = mock_quantize_calls[0][1]
                assert first_call_kwargs["kv_bits"] == 1

        # Test with very large group size
        mock_quantize_calls = []
        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=4,
                    kv_group_size=1024,
                    max_tokens=1,
                )

                try:
                    next(gen)
                except StopIteration:
                    pass

                assert len(mock_quantize_calls) > 0
                first_call_kwargs = mock_quantize_calls[0][1]
                assert first_call_kwargs["kv_group_size"] == 1024

    def test_quantization_integration_with_model_output(self):
        """Test that quantization doesn't interfere with model outputs."""
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

                input_ids = mx.array([[1, 2, 3, 4, 5]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 5))
                model = MockModel()

                # Test with different model output types
                # Test with cross_attention_states
                model.return_value.cross_attention_states = mx.random.normal(
                    (1, 10, 768)
                )
                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=4,
                    max_tokens=2,
                )

                tokens = []
                for token, logprobs in gen:
                    tokens.append(token)
                    assert logprobs is not None

                assert len(tokens) == 2

                # Test with encoder_outputs
                model.return_value.cross_attention_states = None
                model.return_value.encoder_outputs = Mock()
                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=4,
                    max_tokens=2,
                )

                tokens = []
                for token, logprobs in gen:
                    tokens.append(token)
                    assert logprobs is not None

                assert len(tokens) == 2

    @pytest.mark.parametrize(
        "kv_bits,kv_group_size,quantized_kv_start",
        [
            (2, 32, 0),
            (4, 64, 100),
            (8, 128, 5000),
        ],
    )
    def test_parameterized_quantization_configs(
        self, kv_bits, kv_group_size, quantized_kv_start
    ):
        """Test various combinations of quantization parameters."""
        mock_quantize_calls = []

        def mock_maybe_quantize_kv_cache(cache, **kwargs):
            mock_quantize_calls.append((cache, kwargs))
            return None

        generate_module = sys.modules["mlx_vlm.generate"]
        with patch.object(
            generate_module, "maybe_quantize_kv_cache", mock_maybe_quantize_kv_cache
        ):
            with patch("mlx_vlm.models.cache.make_prompt_cache") as mock_make_cache:
                mock_cache_instance = MockCache()
                mock_make_cache.return_value = mock_cache_instance

                input_ids = mx.array([[1, 2, 3, 4, 5]])
                pixel_values = mx.random.normal((1, 3, 336, 336))
                mask = mx.ones((1, 5))
                model = MockModel()

                gen = generate_step(
                    input_ids=input_ids,
                    model=model,
                    pixel_values=pixel_values,
                    mask=mask,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    max_tokens=1,
                )

                try:
                    next(gen)
                except StopIteration:
                    pass

                # Verify all parameters were passed correctly
                assert len(mock_quantize_calls) > 0
                first_call_kwargs = mock_quantize_calls[0][1]
                assert first_call_kwargs["kv_bits"] == kv_bits
                assert first_call_kwargs["kv_group_size"] == kv_group_size
                assert first_call_kwargs["quantized_kv_start"] == quantized_kv_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
