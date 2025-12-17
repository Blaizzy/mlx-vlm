"""Tests for batch generation functionality in mlx_vlm.generate module."""

import sys
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mlx_vlm.generate import (
    Batch,
    BatchGenerationResult,
    BatchGenerator,
    BatchResponse,
    BatchStats,
    GenerationResult,
    _left_pad_prompts,
)

generate_module = sys.modules["mlx_vlm.generate"]

# ============================================================================
# Fixtures and Mock Classes
# ============================================================================


class MockConfig:
    """Mock model config for testing."""

    def __init__(self):
        self.model_type = "test_model"
        self.eos_token_id = [2]
        self.image_token_index = 32000


class MockLanguageModel:
    """Mock language model for testing batch generation."""

    def __init__(self, vocab_size=32000, hidden_size=768):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers = [MagicMock() for _ in range(4)]

    def __call__(self, input_ids, cache=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        seq_len = input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
        logits = mx.random.normal((batch_size, seq_len, self.vocab_size))
        return MagicMock(logits=logits)


class MockModel:
    """Mock VLM model for testing."""

    def __init__(self):
        self.config = MockConfig()
        self.language_model = MockLanguageModel()

    def __call__(self, input_ids, pixel_values=None, cache=None, mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        seq_len = input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
        logits = mx.random.normal((batch_size, seq_len, 32000))
        return MagicMock(
            logits=logits, cross_attention_states=None, encoder_outputs=None
        )

    def get_input_embeddings(self, input_ids, pixel_values, **kwargs):
        batch_size = input_ids.shape[0] if input_ids.ndim > 1 else 1
        seq_len = input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
        return mx.random.normal((batch_size, seq_len, 768))

    def make_cache(self):
        from mlx_vlm.models import cache

        return [cache.KVCache() for _ in range(4)]


class MockStoppingCriteria:
    """Mock stopping criteria."""

    def __init__(self, eos_token_ids=None):
        self.eos_token_ids = eos_token_ids or [2]

    def __call__(self, token):
        return token in self.eos_token_ids

    def add_eos_token_ids(self, tokens):
        if tokens:
            if isinstance(tokens, (list, set)):
                self.eos_token_ids.extend(tokens)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.stopping_criteria = MockStoppingCriteria()

    def decode(self, tokens):
        return f"decoded_{len(tokens)}_tokens"

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class MockDetokenizer:
    """Mock detokenizer for streaming."""

    def __init__(self):
        self.last_segment = ""
        self._tokens = []

    def reset(self):
        self.last_segment = ""
        self._tokens = []

    def add_token(self, token, skip_special_token_ids=None):
        self._tokens.append(token)
        self.last_segment = f"token_{token}"

    def finalize(self):
        self.last_segment = ""


class MockProcessor:
    """Mock processor for testing."""

    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.detokenizer = MockDetokenizer()
        self.image_processor = MagicMock()

    def __call__(
        self, text=None, images=None, audio=None, padding=None, return_tensors="mlx"
    ):
        # Return mock inputs
        batch_size = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": mx.ones((batch_size, 10), dtype=mx.int32),
            "attention_mask": mx.ones((batch_size, 10), dtype=mx.int32),
            "pixel_values": mx.zeros((batch_size, 3, 224, 224)) if images else None,
        }


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_processor():
    return MockProcessor()


# ============================================================================
# Tests for Dataclasses
# ============================================================================


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_default_values(self):
        result = GenerationResult()
        assert result.text == ""
        assert result.token is None
        assert result.logprobs is None
        assert result.prompt_tokens == 0
        assert result.generation_tokens == 0
        assert result.total_tokens == 0
        assert result.prompt_tps == 0.0
        assert result.generation_tps == 0.0
        assert result.peak_memory == 0.0

    def test_with_values(self):
        result = GenerationResult(
            text="Hello world",
            token=42,
            logprobs=[0.1, 0.2, 0.3],
            prompt_tokens=10,
            generation_tokens=5,
            total_tokens=15,
            prompt_tps=100.0,
            generation_tps=50.0,
            peak_memory=2.5,
        )
        assert result.text == "Hello world"
        assert result.token == 42
        assert result.logprobs == [0.1, 0.2, 0.3]
        assert result.prompt_tokens == 10
        assert result.generation_tokens == 5
        assert result.total_tokens == 15
        assert result.prompt_tps == 100.0
        assert result.generation_tps == 50.0
        assert result.peak_memory == 2.5


class TestBatchGenerationResult:
    """Tests for BatchGenerationResult dataclass."""

    def test_creation(self):
        result = BatchGenerationResult(
            texts=["Hello", "World"],
            tokens=[1, 2],
            logprobs=[[0.1], [0.2]],
            prompt_tokens=[10, 12],
            generation_tokens=[5, 6],
            total_tokens=[15, 18],
            prompt_tps=[100.0, 110.0],
            generation_tps=[50.0, 55.0],
            peak_memory=3.0,
            image_sizes=[(224, 224), (336, 336)],
        )
        assert result.texts == ["Hello", "World"]
        assert result.tokens == [1, 2]
        assert result.peak_memory == 3.0
        assert result.image_sizes == [(224, 224), (336, 336)]

    def test_optional_image_sizes(self):
        result = BatchGenerationResult(
            texts=["Hello"],
            tokens=[1],
            logprobs=[[0.1]],
            prompt_tokens=[10],
            generation_tokens=[5],
            total_tokens=[15],
            prompt_tps=[100.0],
            generation_tps=[50.0],
        )
        assert result.image_sizes is None


class TestBatchStats:
    """Tests for BatchStats dataclass."""

    def test_default_values(self):
        stats = BatchStats()
        assert stats.prompt_tokens == 0
        assert stats.prompt_tps == 0
        assert stats.prompt_time == 0
        assert stats.generation_tokens == 0
        assert stats.generation_tps == 0
        assert stats.generation_time == 0
        assert stats.peak_memory == 0

    def test_with_values(self):
        stats = BatchStats(
            prompt_tokens=100,
            prompt_tps=500.0,
            prompt_time=0.2,
            generation_tokens=50,
            generation_tps=250.0,
            generation_time=0.2,
            peak_memory=4.0,
        )
        assert stats.prompt_tokens == 100
        assert stats.prompt_tps == 500.0
        assert stats.generation_tokens == 50


class TestBatchResponse:
    """Tests for BatchResponse dataclass."""

    def test_creation(self):
        stats = BatchStats(prompt_tokens=100)
        response = BatchResponse(
            texts=["Hello", "World"],
            stats=stats,
            image_sizes=[(224, 224), (336, 336)],
        )
        assert response.texts == ["Hello", "World"]
        assert response.stats.prompt_tokens == 100
        assert response.image_sizes == [(224, 224), (336, 336)]

    def test_optional_image_sizes(self):
        stats = BatchStats()
        response = BatchResponse(texts=["Hello"], stats=stats)
        assert response.image_sizes is None


class TestBatch:
    """Tests for Batch dataclass."""

    def test_creation(self):
        batch = Batch(
            uids=[0, 1, 2],
            y=mx.array([10, 20, 30]),
            logprobs=mx.zeros((3, 100)),
            max_tokens=[50, 50, 50],
            num_tokens=[5, 10, 15],
            cache=[MagicMock()],
        )
        assert len(batch) == 3
        assert batch.uids == [0, 1, 2]
        assert batch.max_tokens == [50, 50, 50]

    def test_filter(self):
        # Create mock cache with filter method
        mock_cache = MagicMock()
        mock_cache.filter = MagicMock()

        batch = Batch(
            uids=[0, 1, 2],
            y=mx.array([10, 20, 30]),
            logprobs=mx.zeros((3, 100)),
            max_tokens=[50, 60, 70],
            num_tokens=[5, 10, 15],
            cache=[mock_cache],
        )

        # Keep only indices 0 and 2
        batch.filter([0, 2])

        assert batch.uids == [0, 2]
        assert batch.max_tokens == [50, 70]
        assert batch.num_tokens == [5, 15]
        assert len(batch) == 2

    def test_extend(self):
        mock_cache1 = MagicMock()
        mock_cache1.extend = MagicMock()
        mock_cache2 = MagicMock()

        batch1 = Batch(
            uids=[0, 1],
            y=mx.array([10, 20]),
            logprobs=mx.zeros((2, 100)),
            max_tokens=[50, 50],
            num_tokens=[5, 10],
            cache=[mock_cache1],
        )

        batch2 = Batch(
            uids=[2, 3],
            y=mx.array([30, 40]),
            logprobs=mx.zeros((2, 100)),
            max_tokens=[60, 60],
            num_tokens=[15, 20],
            cache=[mock_cache2],
        )

        batch1.extend(batch2)

        assert batch1.uids == [0, 1, 2, 3]
        assert batch1.max_tokens == [50, 50, 60, 60]
        assert len(batch1) == 4


# ============================================================================
# Tests for Helper Functions
# ============================================================================


class TestLeftPadPrompts:
    """Tests for _left_pad_prompts function."""

    def test_basic_padding(self):
        prompts = [[1, 2, 3], [4, 5], [6]]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (3, 3)
        # Check that shorter prompts are left-padded with zeros
        assert mx.array_equal(padded[0], mx.array([1, 2, 3]))
        assert mx.array_equal(padded[1], mx.array([0, 4, 5]))
        assert mx.array_equal(padded[2], mx.array([0, 0, 6]))

    def test_with_explicit_max_length(self):
        prompts = [[1, 2], [3]]
        padded = _left_pad_prompts(prompts, max_length=5)

        assert padded.shape == (2, 5)
        assert mx.array_equal(padded[0], mx.array([0, 0, 0, 1, 2]))
        assert mx.array_equal(padded[1], mx.array([0, 0, 0, 0, 3]))

    def test_single_prompt(self):
        prompts = [[1, 2, 3, 4, 5]]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (1, 5)
        assert mx.array_equal(padded[0], mx.array([1, 2, 3, 4, 5]))

    def test_equal_length_prompts(self):
        prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (3, 3)
        # No padding needed when all prompts are same length
        assert mx.array_equal(padded[0], mx.array([1, 2, 3]))
        assert mx.array_equal(padded[1], mx.array([4, 5, 6]))
        assert mx.array_equal(padded[2], mx.array([7, 8, 9]))

    def test_empty_prompt(self):
        prompts = [[1, 2], []]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (2, 2)
        assert mx.array_equal(padded[1], mx.array([0, 0]))


# ============================================================================
# Tests for BatchGenerator Class
# ============================================================================


class TestBatchGenerator:
    """Tests for BatchGenerator class."""

    def test_initialization(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=128,
            stop_tokens={2, 3},
        )

        assert gen.max_tokens == 128
        assert gen.model == mock_model.language_model
        assert gen.active_batch is None
        assert gen.uid_count == 0

    def test_insert_prompts(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=100,
        )

        prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        uids = gen.insert(prompts)

        assert len(uids) == 3
        assert uids == [0, 1, 2]
        assert gen.uid_count == 3
        assert len(gen.unprocessed_prompts) == 3

    def test_insert_with_max_tokens(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=50,
        )

        prompts = [[1, 2, 3], [4, 5]]
        max_tokens = [100, 200]
        uids = gen.insert(prompts, max_tokens=max_tokens)

        assert len(uids) == 2
        # Prompts are sorted by length, so check the unprocessed prompts
        assert len(gen.unprocessed_prompts) == 2

    def test_insert_with_single_max_tokens(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=50,
        )

        prompts = [[1, 2, 3], [4, 5]]
        uids = gen.insert(prompts, max_tokens=75)

        assert len(uids) == 2

    def test_stats(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
        )

        # Set some stats manually
        gen._stats.prompt_tokens = 100
        gen._stats.prompt_time = 0.5
        gen._stats.generation_tokens = 50
        gen._stats.generation_time = 0.25

        stats = gen.stats()

        assert stats.prompt_tps == 200.0  # 100 / 0.5
        assert stats.generation_tps == 200.0  # 50 / 0.25

    def test_response_dataclass(self):
        response = BatchGenerator.Response(
            uid=0, token=42, logprobs=mx.array([0.1, 0.2]), finish_reason="stop"
        )

        assert response.uid == 0
        assert response.token == 42
        assert response.finish_reason == "stop"


# ============================================================================
# Tests for batch_generate function
# ============================================================================


class TestBatchGenerate:
    """Tests for the batch_generate function."""

    @patch.object(generate_module, "_generate_batch")
    def test_text_only_batch(self, mock_generate_batch, mock_model, mock_processor):
        """Test batch generation without images."""
        from mlx_vlm.generate import batch_generate

        mock_generate_batch.return_value = (
            ["Response 1", "Response 2"],
            BatchStats(prompt_tokens=20, generation_tokens=10),
        )

        prompts = ["Hello", "World"]
        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=None,
            prompts=prompts,
            max_tokens=50,
        )

        assert isinstance(response, BatchResponse)
        assert response.texts == ["Response 1", "Response 2"]
        mock_generate_batch.assert_called_once()

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_with_images_same_shape(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test batch generation with images of the same shape."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        # Create mock images of the same size
        mock_img1 = MagicMock(spec=Image.Image)
        mock_img1.height = 224
        mock_img1.width = 224

        mock_img2 = MagicMock(spec=Image.Image)
        mock_img2.height = 224
        mock_img2.width = 224

        mock_process_image.side_effect = [mock_img1, mock_img2]
        mock_generate_batch.return_value = (
            ["Response 1", "Response 2"],
            BatchStats(prompt_tokens=40, generation_tokens=20),
        )

        prompts = ["Describe image 1", "Describe image 2"]
        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["path/to/img1.jpg", "path/to/img2.jpg"],
            prompts=prompts,
            max_tokens=50,
        )

        assert isinstance(response, BatchResponse)
        assert len(response.texts) == 2
        # Same shape images should be processed in one batch
        assert mock_generate_batch.call_count == 1

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_with_images_different_shapes(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test batch generation with images of different shapes."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        # Create mock images of different sizes
        mock_img1 = MagicMock(spec=Image.Image)
        mock_img1.height = 224
        mock_img1.width = 224

        mock_img2 = MagicMock(spec=Image.Image)
        mock_img2.height = 336
        mock_img2.width = 336

        mock_img3 = MagicMock(spec=Image.Image)
        mock_img3.height = 224
        mock_img3.width = 224

        mock_process_image.side_effect = [mock_img1, mock_img2, mock_img3]

        # Return correct number of responses for each group:
        # Group 1 (224x224): 2 images (img1, img3) -> 2 responses
        # Group 2 (336x336): 1 image (img2) -> 1 response
        mock_generate_batch.side_effect = [
            (
                ["Response 1", "Response 3"],
                BatchStats(prompt_tokens=20, generation_tokens=10),
            ),
            (["Response 2"], BatchStats(prompt_tokens=10, generation_tokens=5)),
        ]

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
            prompts=prompts,
            max_tokens=50,
            group_by_shape=True,
        )

        assert isinstance(response, BatchResponse)
        # Different shapes should result in multiple batches (2 groups: 224x224 and 336x336)
        assert mock_generate_batch.call_count == 2
        # All 3 responses should be present
        assert len(response.texts) == 3

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_track_image_sizes(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test that image sizes are tracked correctly."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        mock_img = MagicMock(spec=Image.Image)
        mock_img.height = 512
        mock_img.width = 384

        mock_process_image.return_value = mock_img
        mock_generate_batch.return_value = (
            ["Response"],
            BatchStats(prompt_tokens=20, generation_tokens=10),
        )

        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["test.jpg"],
            prompts=["Describe this"],
            track_image_sizes=True,
        )

        assert response.image_sizes is not None
        assert response.image_sizes[0] == (512, 384)

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_disable_track_image_sizes(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test that image sizes tracking can be disabled."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        mock_img = MagicMock(spec=Image.Image)
        mock_img.height = 512
        mock_img.width = 384

        mock_process_image.return_value = mock_img
        mock_generate_batch.return_value = (
            ["Response"],
            BatchStats(prompt_tokens=20, generation_tokens=10),
        )

        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["test.jpg"],
            prompts=["Describe this"],
            track_image_sizes=False,
        )

        assert response.image_sizes is None

    @patch.object(generate_module, "_generate_batch")
    def test_per_sample_max_tokens(
        self, mock_generate_batch, mock_model, mock_processor
    ):
        """Test batch generation with per-sample max_tokens."""
        from mlx_vlm.generate import batch_generate

        mock_generate_batch.return_value = (
            ["Short", "Long response here"],
            BatchStats(),
        )

        prompts = ["Short prompt", "Longer prompt here"]
        max_tokens = [50, 200]

        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=None,
            prompts=prompts,
            max_tokens=max_tokens,
        )

        assert isinstance(response, BatchResponse)
        assert len(response.texts) == 2

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_single_image_string(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test that a single image string is converted to list."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        mock_img = MagicMock(spec=Image.Image)
        mock_img.height = 224
        mock_img.width = 224

        mock_process_image.return_value = mock_img
        mock_generate_batch.return_value = (["Response"], BatchStats())

        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images="single_image.jpg",  # String, not list
            prompts=["Describe this"],
        )

        assert isinstance(response, BatchResponse)
        mock_process_image.assert_called_once()

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_verbose_output(
        self,
        mock_process_image,
        mock_generate_batch,
        mock_model,
        mock_processor,
        capsys,
    ):
        """Test verbose output in batch generation."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        mock_img = MagicMock(spec=Image.Image)
        mock_img.height = 224
        mock_img.width = 224

        mock_process_image.return_value = mock_img
        mock_generate_batch.return_value = (
            ["Response"],
            BatchStats(
                prompt_tokens=100,
                prompt_time=0.1,
                generation_tokens=50,
                generation_time=0.2,
            ),
        )

        batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["test.jpg"],
            prompts=["Describe this"],
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "[batch_generate]" in captured.out

    @patch.object(generate_module, "_generate_batch")
    @patch("mlx_vlm.utils.process_image")
    def test_disable_grouping(
        self, mock_process_image, mock_generate_batch, mock_model, mock_processor
    ):
        """Test batch generation with grouping disabled."""
        from PIL import Image

        from mlx_vlm.generate import batch_generate

        # Create mock images of different sizes
        mock_img1 = MagicMock(spec=Image.Image)
        mock_img1.height = 224
        mock_img1.width = 224

        mock_img2 = MagicMock(spec=Image.Image)
        mock_img2.height = 336
        mock_img2.width = 336

        mock_process_image.side_effect = [mock_img1, mock_img2]
        # When grouping is disabled, both images are in one group
        mock_generate_batch.return_value = (["Response 1", "Response 2"], BatchStats())

        response = batch_generate(
            model=mock_model,
            processor=mock_processor,
            images=["img1.jpg", "img2.jpg"],
            prompts=["Prompt 1", "Prompt 2"],
            group_by_shape=False,  # Disable grouping
        )

        assert isinstance(response, BatchResponse)
        assert len(response.texts) == 2


# ============================================================================
# Tests for stats aggregation
# ============================================================================


class TestBatchStatsAggregation:
    """Tests for stats aggregation in batch generation."""

    def test_stats_accumulation(self):
        """Test that stats are properly accumulated across batches."""
        total_stats = BatchStats()

        # Simulate processing multiple batches
        batch_stats = [
            BatchStats(
                prompt_tokens=100,
                prompt_time=0.1,
                generation_tokens=50,
                generation_time=0.2,
            ),
            BatchStats(
                prompt_tokens=150,
                prompt_time=0.15,
                generation_tokens=75,
                generation_time=0.3,
            ),
        ]

        for stats in batch_stats:
            total_stats.prompt_tokens += stats.prompt_tokens
            total_stats.prompt_time += stats.prompt_time
            total_stats.generation_tokens += stats.generation_tokens
            total_stats.generation_time += stats.generation_time

        assert total_stats.prompt_tokens == 250
        assert total_stats.prompt_time == pytest.approx(0.25)
        assert total_stats.generation_tokens == 125
        assert total_stats.generation_time == pytest.approx(0.5)

        # Calculate TPS
        if total_stats.prompt_time > 0:
            total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
        if total_stats.generation_time > 0:
            total_stats.generation_tps = (
                total_stats.generation_tokens / total_stats.generation_time
            )

        assert total_stats.prompt_tps == pytest.approx(1000.0)
        assert total_stats.generation_tps == pytest.approx(250.0)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in batch generation."""

    def test_empty_prompts_list(self, mock_model, mock_processor):
        """Test behavior with empty prompts list."""
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
        )

        uids = gen.insert([])
        assert uids == []
        assert gen.uid_count == 0

    def test_single_token_prompt(self):
        """Test left padding with single token prompts."""
        prompts = [[1], [2], [3]]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (3, 1)
        assert mx.array_equal(padded[0], mx.array([1]))

    def test_very_long_prompt(self):
        """Test left padding with very long prompts."""
        short_prompt = [1, 2]
        long_prompt = list(range(1000))

        padded = _left_pad_prompts([short_prompt, long_prompt])

        assert padded.shape == (2, 1000)
        # First prompt should have 998 padding tokens
        assert padded[0, 0].item() == 0
        assert padded[0, -1].item() == 2

    def test_batch_response_with_empty_texts(self):
        """Test BatchResponse with empty texts."""
        stats = BatchStats()
        response = BatchResponse(texts=[], stats=stats)

        assert response.texts == []
        assert response.image_sizes is None


# ============================================================================
# Tests for Thinking Budget
# ============================================================================


class TestThinkingBudget:
    """Tests for thinking budget enforcement in generate_step."""

    def test_thinking_budget_enforcement(self):
        """Test that thinking budget correctly limits tokens in think blocks."""
        from mlx_vlm.generate import generate_step

        # Token IDs for testing
        THINK_START_ID = 100  # <think>
        THINK_END_ID = 101  # </think>
        REGULAR_ID = 50  # regular token

        # Create a mock model that returns predictable tokens
        class ThinkingMockModel:
            def __init__(self):
                self.call_count = 0
                self.config = MockConfig()
                # Sequence: <think>, regular, regular, ... (thinking never ends naturally)
                self.token_sequence = [THINK_START_ID] + [REGULAR_ID] * 20

            @property
            def language_model(self):
                return self

            def make_cache(self):
                from mlx_vlm.models import cache

                return [cache.KVCache() for _ in range(4)]

            def __call__(
                self, input_ids, pixel_values=None, cache=None, mask=None, **kwargs
            ):
                # Return logits that favor the next token in sequence
                batch_size = 1
                seq_len = (
                    input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
                )
                logits = mx.zeros((batch_size, seq_len, 32000))

                # Set high logit for the token we want to generate
                if self.call_count < len(self.token_sequence):
                    target_token = self.token_sequence[self.call_count]
                else:
                    target_token = REGULAR_ID
                logits[:, -1, target_token] = 100.0
                self.call_count += 1

                return MagicMock(
                    logits=logits,
                    cross_attention_states=None,
                    encoder_outputs=None,
                )

        model = ThinkingMockModel()
        input_ids = mx.array([[1, 2, 3]])
        pixel_values = None
        mask = None

        # Generate with thinking_budget=5
        tokens = []
        for token, logprobs in generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            max_tokens=15,
            temperature=0,
            thinking_budget=5,
            thinking_start_token_id=THINK_START_ID,
            thinking_end_token_id=THINK_END_ID,
        ):
            tokens.append(token)

        # Verify: should see <think>, then 5 thinking tokens, then </think> forced
        assert tokens[0] == THINK_START_ID  # First token is <think>
        assert THINK_END_ID in tokens  # </think> should be forced

        # Find where </think> appears
        think_end_idx = tokens.index(THINK_END_ID)
        # Budget of 5 means: tokens at indices 1-5 are thinking tokens (5 total),
        # then </think> is forced at index 6
        # Token sequence: <think>, tok, tok, tok, tok, tok, </think>
        assert think_end_idx == 6

    def test_thinking_budget_with_natural_end(self):
        """Test that natural </think> tokens are respected before budget is hit."""
        from mlx_vlm.generate import generate_step

        # Token IDs for testing
        THINK_START_ID = 100  # <think>
        THINK_END_ID = 101  # </think>
        REGULAR_ID = 50  # regular token

        class NaturalEndMockModel:
            def __init__(self):
                self.call_count = 0
                self.config = MockConfig()
                # Sequence: <think>, regular, regular, </think> (natural end at 3 tokens)
                self.token_sequence = [
                    THINK_START_ID,
                    REGULAR_ID,
                    REGULAR_ID,
                    THINK_END_ID,
                    REGULAR_ID,
                    REGULAR_ID,
                ]

            @property
            def language_model(self):
                return self

            def make_cache(self):
                from mlx_vlm.models import cache

                return [cache.KVCache() for _ in range(4)]

            def __call__(
                self, input_ids, pixel_values=None, cache=None, mask=None, **kwargs
            ):
                batch_size = 1
                seq_len = (
                    input_ids.shape[-1] if input_ids.ndim > 1 else input_ids.shape[0]
                )
                logits = mx.zeros((batch_size, seq_len, 32000))

                if self.call_count < len(self.token_sequence):
                    target_token = self.token_sequence[self.call_count]
                else:
                    target_token = REGULAR_ID
                logits[:, -1, target_token] = 100.0
                self.call_count += 1

                return MagicMock(
                    logits=logits,
                    cross_attention_states=None,
                    encoder_outputs=None,
                )

        model = NaturalEndMockModel()
        input_ids = mx.array([[1, 2, 3]])

        # Generate with thinking_budget=10 (higher than natural end)
        tokens = []
        for token, logprobs in generate_step(
            input_ids,
            model,
            None,
            None,
            max_tokens=10,
            temperature=0,
            thinking_budget=10,
            thinking_start_token_id=THINK_START_ID,
            thinking_end_token_id=THINK_END_ID,
        ):
            tokens.append(token)

        # Natural </think> should appear at index 3 (not forced at budget)
        assert tokens[0] == THINK_START_ID
        think_end_idx = tokens.index(THINK_END_ID)
        assert think_end_idx == 3  # Natural end, not forced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
