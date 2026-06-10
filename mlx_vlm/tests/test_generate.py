"""Tests for batch generation functionality in mlx_vlm.generate module."""

import logging
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache, KVCache

from mlx_vlm import apc as apc_module
from mlx_vlm.generate import (
    BatchGenerationResult,
    BatchGenerator,
    BatchResponse,
    BatchStats,
    GenerationBatch,
    GenerationResult,
    PromptProcessingBatch,
    SpeculativeGenerationBatch,
    _left_pad_prompts,
    _prime_cached_prefix_rope_state,
)
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate import dispatch as dispatch_module
from mlx_vlm.generate import normalize_resize_shape
from mlx_vlm.utils import ThinkingBudgetCriteria

generate_module = sys.modules["mlx_vlm.generate"]
image_module = __import__("mlx_vlm.generate.image", fromlist=[""])

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


class TestGenerationBatch:
    """Tests for GenerationBatch class."""

    def test_empty_creation(self):
        mock_model = MagicMock()
        sampler = lambda x: mx.argmax(x, axis=-1)
        stop_criteria = lambda tok: tok == 2
        batch = GenerationBatch.empty(mock_model, sampler, stop_criteria)
        assert len(batch) == 0
        assert batch.uids == []
        assert batch.max_tokens == []

    def test_filter(self):
        mock_model = MagicMock()
        sampler = lambda x: mx.argmax(x, axis=-1)
        stop_criteria = lambda tok: tok == 2
        batch = GenerationBatch.empty(mock_model, sampler, stop_criteria)
        batch.uids = [0, 1, 2]
        batch.max_tokens = [50, 60, 70]
        batch._num_tokens = [5, 10, 15]
        batch._next_tokens = mx.array([10, 20, 30])
        batch._next_logprobs = mx.zeros((3, 100))

        # Keep only indices 0 and 2
        batch.filter([0, 2])

        assert batch.uids == [0, 2]
        assert batch.max_tokens == [50, 70]
        assert batch._num_tokens == [5, 15]
        assert len(batch) == 2

    @staticmethod
    def _mrope_batch(uids, deltas):
        sampler = lambda x: mx.argmax(x, axis=-1)
        batch = GenerationBatch.empty(MagicMock(), sampler, lambda tok: False)
        batch.uids = list(uids)
        batch.max_tokens = [10] * len(uids)
        batch._num_tokens = [0] * len(uids)
        batch._rope_deltas = mx.array(deltas, dtype=mx.int32)
        return batch

    def test_extend_concatenates_and_filters_per_row_rope_deltas(self):
        a = self._mrope_batch([0, 1], [[5], [7]])
        b = self._mrope_batch([2], [[0]])
        a.extend(b)
        assert a._rope_deltas.tolist() == [[5], [7], [0]]
        a.filter([0, 2])
        assert a.uids == [0, 2]
        assert a._rope_deltas.tolist() == [[5], [0]]

    def test_extend_rejects_mixed_mrope_state(self):
        a = self._mrope_batch([0], [[3]])
        b = self._mrope_batch([1], [[0]])
        b._rope_deltas = None
        with pytest.raises(RuntimeError, match="MRoPE"):
            a.extend(b)

    def test_extend_into_empty_accumulator_absorbs_mrope_state(self):
        empty = GenerationBatch.empty(
            MagicMock(), lambda x: mx.argmax(x, axis=-1), lambda tok: False
        )
        empty.extend(self._mrope_batch([0, 1], [[5], [7]]))
        assert empty._rope_deltas.tolist() == [[5], [7]]

    def test_extend_materializes_pending_decode_before_cache_merge(self, monkeypatch):
        calls = []

        class RecordingCache:
            @property
            def state(self):
                return ()

            def extend(self, other):
                calls.append(("extend-cache",))

        def record_eval(batch):
            calls.append(("eval", tuple(batch.uids)))

        monkeypatch.setattr(GenerationBatch, "_eval_pending_state", record_eval)

        a = self._mrope_batch([0], [[0]])
        b = self._mrope_batch([1], [[5]])
        a.prompt_cache = [RecordingCache()]
        b.prompt_cache = [RecordingCache()]

        a.extend(b)

        assert calls == [("eval", (0,)), ("eval", (1,)), ("extend-cache",)]

    def test_filter_materializes_pending_decode_before_cache_filter(self, monkeypatch):
        calls = []

        class RecordingCache:
            @property
            def state(self):
                return ()

            def filter(self, keep):
                calls.append(("filter-cache", keep.tolist()))

        def record_eval(batch):
            calls.append(("eval", tuple(batch.uids)))

        monkeypatch.setattr(GenerationBatch, "_eval_pending_state", record_eval)

        batch = self._mrope_batch([0, 1], [[0], [5]])
        batch.prompt_cache = [RecordingCache()]
        batch._next_tokens = mx.array([10, 20], dtype=mx.int32)

        batch.filter([0])

        assert calls == [("eval", (0, 1)), ("filter-cache", [0])]

    @staticmethod
    def _capture(value, B):
        from mlx_vlm.generate import PromptProcessingBatch

        return PromptProcessingBatch._capture_rope_deltas(
            SimpleNamespace(_rope_deltas=value), B
        )

    def test_capture_rope_deltas(self):
        from mlx_vlm.generate import PromptProcessingBatch

        assert PromptProcessingBatch._capture_rope_deltas(SimpleNamespace(), 3) is None
        assert self._capture(None, 3).tolist() == [[0], [0], [0]]
        assert self._capture(mx.array([[5], [7], [9]], dtype=mx.int32), 3).tolist() == [
            [5],
            [7],
            [9],
        ]
        # Falcon OCR singleton: (1, 1) broadcasts to (B, 1).
        assert self._capture(mx.array([[5]], dtype=mx.int32), 4).tolist() == [[5]] * 4
        assert self._capture(mx.array([[5], [7]], dtype=mx.int32), 3).tolist() == [
            [5],
            [7],
            [7],
        ]
        assert self._capture(mx.array([[5], [7], [9]], dtype=mx.int32), 2).tolist() == [
            [5],
            [7],
        ]


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
        assert len(gen._generation_batch) == 0
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

        # Set some stats manually via counters
        gen._prompt_tokens_counter = 100
        gen._prompt_time_counter = 0.5
        gen._gen_tokens_counter = 50

        stats = gen.stats()

        assert stats.prompt_tps == 200.0  # 100 / 0.5
        assert stats.prompt_tokens == 100

    def test_next_reports_prompt_progress_for_completed_prefill(
        self, mock_model, mock_processor, monkeypatch
    ):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            prefill_batch_size=1,
            completion_batch_size=1,
            prefill_step_size=None,
        )
        prompt = [1, 2, 3]
        inputs_embeds = mx.random.normal((1, len(prompt), 8))
        uids = gen.insert(
            [prompt],
            prompt_kwargs=[{"inputs_embeds": inputs_embeds}],
        )
        ticks = iter([10.0, 10.2])
        monkeypatch.setattr(ar_module.time, "perf_counter", lambda: next(ticks))

        prompt_responses, generation_responses = gen.next()

        assert generation_responses == []
        assert len(prompt_responses) == 1
        assert prompt_responses[0].uid == uids[0]
        assert prompt_responses[0].prompt_tokens == len(prompt)
        assert prompt_responses[0].prompt_tps == pytest.approx(15.0)
        assert prompt_responses[0].prompt_time == pytest.approx(0.2)
        assert prompt_responses[0].cached_tokens == 0

    def test_prompt_progress_reports_apc_cached_tokens(self):
        batch = PromptProcessingBatch(
            model=SimpleNamespace(),
            uids=[1, 2],
            input_ids=[[4, 5], [6, 7, 8]],
            max_tokens=[1, 1],
            inputs_embeds=mx.ones((2, 3, 4)),
            prompt_kwargs={},
            prefill_step_size=None,
            warm_cache=[],
            apc_meta=[
                {"full_input_ids": [1, 2, 3, 4, 5], "prefix_len": 3},
                None,
            ],
        )
        batch.record_prompt_time(0.5)

        progress = batch.prompt_progress()

        assert [p.prompt_tokens for p in progress] == [5, 3]
        assert [p.cached_tokens for p in progress] == [3, 0]

    def test_response_dataclass(self):
        response = GenerationBatch.Response(
            uid=0, token=42, token_logprob=-0.5, finish_reason="stop"
        )

        assert response.uid == 0
        assert response.token == 42
        assert response.finish_reason == "stop"

    def test_generation_batch_applies_per_sequence_logits_processors(self):
        class FixedLogitModel:
            def __call__(self, input_ids, cache=None, **kwargs):
                token_scores = mx.array([0.0, 10.0, 0.0, 0.0])
                logits = mx.broadcast_to(
                    token_scores, (input_ids.shape[0], input_ids.shape[1], 4)
                )
                return MagicMock(logits=logits)

        seen_contexts = []

        def force_token_2(tokens, logits):
            seen_contexts.append(tokens.tolist())
            token_scores = mx.array([-1e9, -1e9, 0.0, -1e9])
            return mx.broadcast_to(token_scores, logits.shape)

        batch = GenerationBatch(
            model=FixedLogitModel(),
            uids=[0, 1],
            inputs=mx.array([5, 6], dtype=mx.int32),
            prompt_cache=[],
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[2, 2],
            token_context=[mx.array([10]), mx.array([20])],
            logits_processors=[[force_token_2], [force_token_2]],
        )

        first = batch.next()
        assert [r.token for r in first] == [5, 6]
        assert seen_contexts == [[10, 5], [20, 6]]

        second = batch.next()
        assert [r.token for r in second] == [2, 2]

    def test_generation_batch_thinking_budget_criteria_can_force_next_token(self):
        class FixedLogitModel:
            def __call__(self, input_ids, cache=None, **kwargs):
                token_scores = mx.array([0.0, 10.0, 0.0, 0.0])
                logits = mx.broadcast_to(
                    token_scores, (input_ids.shape[0], input_ids.shape[1], 4)
                )
                return MagicMock(logits=logits)

        class ForceAfterFirst:
            def __init__(self):
                self.forced_token_id = None

            def __call__(self, token):
                self.forced_token_id = 3 if token == 5 else None

            def apply_forced_token(self, next_y):
                if self.forced_token_id is None:
                    return next_y
                forced = mx.array([self.forced_token_id], dtype=mx.int32)
                self.forced_token_id = None
                return forced

        batch = GenerationBatch(
            model=FixedLogitModel(),
            uids=[0],
            inputs=mx.array([5], dtype=mx.int32),
            prompt_cache=[],
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[2],
            thinking_budget_criteria=[ForceAfterFirst()],
        )

        first = batch.next()
        assert [r.token for r in first] == [5]

        second = batch.next()
        assert [r.token for r in second] == [3]

    def test_generation_batch_uses_greedy_hidden_argmax_without_logprobs(self):
        class FastArgmaxModel:
            def __init__(self):
                self.calls = []

            def __call__(self, input_ids, cache=None, **kwargs):
                del cache
                self.calls.append(kwargs)
                assert kwargs["return_hidden"] is True
                assert kwargs["skip_logits"] is True
                hidden = mx.ones((input_ids.shape[0], input_ids.shape[1], 3))
                return SimpleNamespace(hidden_states=[hidden])

            def speculative_argmax_from_hidden(self, hidden):
                return mx.full((hidden.shape[0], hidden.shape[1]), 7, dtype=mx.int32)

        model = FastArgmaxModel()
        batch = GenerationBatch(
            model=model,
            uids=[0, 1],
            inputs=mx.array([5, 6], dtype=mx.int32),
            prompt_cache=[],
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[2, 2],
            greedy_sampling=True,
        )
        batch.compute_logprobs = False

        first = batch.next()
        assert [r.token for r in first] == [5, 6]
        assert batch._next_tokens.tolist() == [7, 7]
        assert model.calls == [{"return_hidden": True, "skip_logits": True}]

    def test_speculative_generation_batch_drains_full_round(self, monkeypatch):
        def fake_rounds(*args, **kwargs):
            del args, kwargs
            yield [1, 10], {"round_pos": 0, "round_len": 2}
            yield [2, 11], {"round_pos": 1, "round_len": 2}
            yield [3, 12], {"round_pos": 0, "round_len": 1}

        monkeypatch.setattr(ar_module, "run_speculative_server_rounds", fake_rounds)

        batch = SpeculativeGenerationBatch(
            model=SimpleNamespace(),
            draft_model=SimpleNamespace(),
            draft_kind="mtp",
            uids=[100, 200],
            first_tokens=mx.array([0, 9], dtype=mx.int32),
            prompt_cache=[],
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[10, 10],
            hidden=mx.zeros((2, 1, 1)),
            shared_kv_states=None,
            prompt_tokens=mx.array([[0], [9]], dtype=mx.int32),
        )

        first = batch.next()
        assert [(r.uid, r.token) for r in first] == [(100, 0), (200, 9)]

        second = batch.next()
        assert [(r.uid, r.token) for r in second] == [
            (100, 1),
            (200, 10),
            (100, 2),
            (200, 11),
        ]

    def test_generation_batch_extend_keeps_processor_context_aligned(self):
        class FixedLogitModel:
            def __call__(self, input_ids, cache=None, **kwargs):
                token_scores = mx.array([0.0, 10.0, 0.0, 0.0])
                logits = mx.broadcast_to(
                    token_scores, (input_ids.shape[0], input_ids.shape[1], 4)
                )
                return MagicMock(logits=logits)

        seen_contexts = []

        def force_token_2(tokens, logits):
            seen_contexts.append(tokens.tolist())
            token_scores = mx.array([-1e9, -1e9, 0.0, -1e9])
            return mx.broadcast_to(token_scores, logits.shape)

        sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        stop_criteria = lambda token: False
        plain = GenerationBatch(
            model=FixedLogitModel(),
            uids=[0, 1],
            inputs=mx.array([5, 6], dtype=mx.int32),
            prompt_cache=[],
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=[2, 2],
            logits_processors=[None, None],
        )
        structured = GenerationBatch(
            model=FixedLogitModel(),
            uids=[2],
            inputs=mx.array([7], dtype=mx.int32),
            prompt_cache=[],
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=[2],
            token_context=[[30]],
            logits_processors=[[force_token_2]],
        )

        plain.extend(structured)
        assert plain.token_context == [[], [], [30]]

        first = plain.next()
        assert [r.token for r in first] == [5, 6, 7]
        assert seen_contexts == [[30, 7]]

    def test_generation_batch_extend_promotes_singleton_kv_cache(self):
        def make_kv_cache(value):
            c = KVCache()
            keys = mx.full((1, 2, 3, 4), value, dtype=mx.float32)
            values = mx.full((1, 2, 3, 4), value + 1, dtype=mx.float32)
            c.update_and_fetch(keys, values)
            return c

        sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        stop_criteria = lambda token: False
        first = GenerationBatch(
            model=MagicMock(),
            uids=[0],
            inputs=mx.array([5], dtype=mx.int32),
            prompt_cache=[make_kv_cache(1.0)],
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=[2],
        )
        second = GenerationBatch(
            model=MagicMock(),
            uids=[1],
            inputs=mx.array([6], dtype=mx.int32),
            prompt_cache=[make_kv_cache(3.0)],
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=[2],
        )

        first.extend(second)

        assert isinstance(first.prompt_cache[0], BatchKVCache)
        assert first.prompt_cache[0].left_padding.tolist() == [0, 0]
        assert first.prompt_cache[0].keys.shape[0] == 2
        assert first._next_tokens.tolist() == [5, 6]

    def test_remove_from_unprocessed(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=50,
        )
        uids = gen.insert([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert len(gen.unprocessed_prompts) == 3

        assert gen.remove(uids[1]) is True
        assert len(gen.unprocessed_prompts) == 2
        remaining_uids = [seq[0] for seq in gen.unprocessed_prompts]
        assert uids[1] not in remaining_uids
        assert uids[0] in remaining_uids
        assert uids[2] in remaining_uids

    def test_remove_missing_uid_returns_false(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            max_tokens=50,
        )
        gen.insert([[1, 2, 3]])
        assert gen.remove(9999) is False


# ============================================================================
# Tests for batch_generate function
# ============================================================================


class TestBatchGenerate:
    """Tests for the batch_generate function."""

    @patch.object(ar_module, "_generate_batch")
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

    def test_generate_batch_splits_batched_prompt_kwargs_per_row(
        self, mock_model, mock_processor
    ):
        """Regression test for Gemma 4-style batched ``inputs_embeds``."""

        class _EmbeddingOutput:
            def __init__(self, inputs_embeds, position_ids):
                self.inputs_embeds = inputs_embeds
                self.position_ids = position_ids

            def to_dict(self):
                return {
                    "inputs_embeds": self.inputs_embeds,
                    "position_ids": self.position_ids,
                }

        class _StopInsert(Exception):
            pass

        batch_size = 3
        seq_len = 5
        hidden_size = 7
        input_ids = mx.array(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
            ],
            dtype=mx.int32,
        )
        inputs_embeds = mx.arange(
            batch_size * seq_len * hidden_size, dtype=mx.float32
        ).reshape(batch_size, seq_len, hidden_size)
        position_ids = mx.arange(batch_size * seq_len, dtype=mx.int32).reshape(
            batch_size, seq_len
        )
        embedding_output = _EmbeddingOutput(inputs_embeds, position_ids)

        def fake_insert(
            self, prompts, max_tokens, prompt_kwargs=None, logits_processors=None
        ):
            assert len(prompts) == batch_size
            assert len(prompt_kwargs) == batch_size
            for i, kw in enumerate(prompt_kwargs):
                assert kw["inputs_embeds"].shape == (1, seq_len, hidden_size)
                assert kw["position_ids"].shape == (1, seq_len)
                assert kw["inputs_embeds"].tolist() == inputs_embeds[i : i + 1].tolist()
                assert kw["position_ids"].tolist() == position_ids[i : i + 1].tolist()
            raise _StopInsert

        with (
            patch.object(
                ar_module,
                "apply_chat_template",
                side_effect=lambda processor, config, prompt, num_images=0: prompt,
            ),
            patch.object(
                ar_module,
                "prepare_inputs",
                return_value={
                    "input_ids": input_ids,
                    "attention_mask": mx.ones((batch_size, seq_len), dtype=mx.int32),
                },
            ),
            patch.object(
                mock_model, "get_input_embeddings", return_value=embedding_output
            ),
            patch.object(ar_module.BatchGenerator, "insert", new=fake_insert),
        ):
            with pytest.raises(_StopInsert):
                ar_module._generate_batch(
                    mock_model,
                    mock_processor,
                    prompts=["alpha", "beta", "gamma"],
                    max_tokens=5,
                )

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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

    @patch.object(ar_module, "_generate_batch")
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
# Tests for ThinkingBudgetCriteria
# ============================================================================


class FakeTokenizer:
    """Mock tokenizer that maps token strings to fixed IDs."""

    TOKEN_MAP = {"<think>": 99, "</think>": 100, "\n": 10}

    def encode(self, text, add_special_tokens=False):
        if text in self.TOKEN_MAP:
            return [self.TOKEN_MAP[text]]
        return [0]


class TestThinkingBudgetCriteria:
    """Tests for ThinkingBudgetCriteria class."""

    def test_thinking_model(self):
        """Test thinking budget for thinking models (enable_thinking=True)."""
        criteria = ThinkingBudgetCriteria(
            tokenizer=FakeTokenizer(),
            thinking_budget=5,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=True,
        )

        # enable_thinking=True — already in thinking mode
        assert criteria.in_thinking is True

        # Tokens within budget return None
        for i in range(5):
            assert criteria(50 + i) is None
        assert criteria.thinking_token_count == 5
        assert criteria.budget_exceeded is False

        # Exceeding budget forces \n then </think>
        assert criteria(60) == 10  # \n
        assert criteria(60) == 100  # </think>
        assert criteria.budget_exceeded is True

        # End token resets state
        assert criteria(100) is None
        assert criteria.in_thinking is False
        assert criteria.budget_exceeded is False

    def test_non_thinking_model(self):
        """Test thinking budget for non-thinking models (enable_thinking=False)."""
        criteria = ThinkingBudgetCriteria(
            tokenizer=FakeTokenizer(),
            thinking_budget=3,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=False,
        )

        # Not in thinking initially
        assert criteria.in_thinking is False

        # Tokens are not counted — model is not in thinking mode
        criteria(50)
        criteria(51)
        assert criteria.thinking_token_count == 0

        # Start token does NOT enter thinking mode when enable_thinking=False
        assert criteria(99) is None
        assert criteria.in_thinking is False

        # Tokens still not counted
        for i in range(3):
            assert criteria(50 + i) is None
        assert criteria.thinking_token_count == 0
        assert criteria.budget_exceeded is False

    def _make_criteria(self, enable_thinking=True):
        return ThinkingBudgetCriteria(
            tokenizer=FakeTokenizer(),
            thinking_budget=5,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=enable_thinking,
        )

    def test_apply_forced_token_safe_before_first_call(self):
        """Regression: apply_forced_token must be safe before __call__ ever runs.

        forced_token_id has to be initialised in __init__; otherwise the first
        apply_forced_token (e.g. on the very first decode step) raises
        AttributeError. This crashed real generations on Gemma-style models
        whose first generated token is the thinking delimiter.
        """
        criteria = self._make_criteria()
        y = mx.array([7])
        # No __call__ yet: must be a no-op that returns the input unchanged.
        assert criteria.apply_forced_token(y).tolist() == [7]

    def test_apply_forced_token_safe_after_start_delimiter(self):
        """Regression: the start-token early return in __call__ does not set
        forced_token_id, so apply_forced_token must remain safe afterwards."""
        criteria = self._make_criteria()
        # First generated token is the start delimiter -> early return, no force.
        assert criteria(99) is None
        assert criteria.apply_forced_token(mx.array([7])).tolist() == [7]

    def test_apply_forced_token_safe_after_end_delimiter(self):
        """Regression: the end-token early return in __call__ does not set
        forced_token_id, so apply_forced_token must remain safe afterwards."""
        criteria = self._make_criteria()
        # End delimiter resets thinking state and returns None without forcing.
        assert criteria(100) is None
        assert criteria.apply_forced_token(mx.array([8])).tolist() == [8]

    def test_apply_forced_token_emits_forced_token_when_budget_exceeded(self):
        """End-to-end: once the budget is exceeded, the token returned by
        __call__ is the same one apply_forced_token injects into the stream."""
        criteria = self._make_criteria()
        # Burn the budget (5 tokens), then trip it on the 6th.
        for i in range(5):
            assert criteria(50 + i) is None
        forced = criteria(60)  # \n forced
        assert forced == 10
        assert criteria.apply_forced_token(mx.array([0])).tolist() == [10]


class TestSamplerArgs:
    """Tests for sampler argument forwarding."""

    @patch.object(generate_module.cache, "make_prompt_cache", return_value=[])
    @patch.object(generate_module, "make_logits_processors", return_value=[])
    @patch.object(generate_module, "make_sampler")
    def test_generate_step_passes_sampling_and_logits_processor_args(
        self,
        mock_make_sampler,
        mock_make_logits_processors,
        _mock_prompt_cache,
    ):
        mock_make_sampler.return_value = lambda logprobs: mx.array([0])

        model = MagicMock()
        model.language_model.return_value = MagicMock(
            logits=mx.zeros((1, 1, 4)),
            cross_attention_states=None,
            encoder_outputs=None,
        )

        embedding_output = MagicMock()
        embedding_output.inputs_embeds = mx.zeros((1, 1, 4))
        embedding_output.to_dict.return_value = {}
        model.get_input_embeddings.return_value = embedding_output

        gen = generate_module.generate_step(
            input_ids=mx.array([[1]], dtype=mx.int32),
            model=model,
            pixel_values=None,
            mask=None,
            max_tokens=1,
            temperature=0.7,
            top_p=0.9,
            min_p=0.05,
            top_k=32,
            repetition_penalty=1.15,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={3: -0.75},
        )

        next(gen)

        mock_make_sampler.assert_called_once_with(
            temp=0.7,
            top_p=0.9,
            min_p=0.05,
            top_k=32,
        )
        mock_make_logits_processors.assert_called_once_with(
            {3: -0.75}, 1.15, 512, 0.2, 256, 0.3, 128
        )


@pytest.mark.parametrize(("verbose", "disabled"), [(False, True), (True, False)])
def test_generate_step_prefill_tqdm_respects_verbose(verbose, disabled):
    pbar = MagicMock()

    model = MagicMock()
    model.language_model.return_value = MagicMock(
        logits=mx.zeros((1, 1, 4)),
        cross_attention_states=None,
        encoder_outputs=None,
    )
    model.no_chunked_prefill = False

    embedding_output = MagicMock()
    embedding_output.inputs_embeds = mx.zeros((1, 5, 4))
    embedding_output.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding_output

    with (
        patch.object(generate_module.cache, "make_prompt_cache", return_value=[]),
        patch.object(generate_module, "make_logits_processors", return_value=[]),
        patch.object(
            generate_module, "make_sampler", return_value=lambda _: mx.array([0])
        ),
        patch.object(ar_module, "tqdm") as mock_tqdm,
    ):
        mock_tqdm.return_value.__enter__.return_value = pbar

        gen = generate_module.generate_step(
            input_ids=mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
            model=model,
            pixel_values=None,
            mask=None,
            max_tokens=1,
            prefill_step_size=2,
            verbose=verbose,
        )

        next(gen)

    mock_tqdm.assert_called_once()
    assert mock_tqdm.call_args.kwargs["disable"] is disabled
    assert pbar.update.call_count > 0


def test_generate_step_chunks_prefill_when_model_policy_allows_speculation():
    model = MagicMock()
    model.no_chunked_prefill = False
    model.chunked_prefill_policy.return_value = True

    output = SimpleNamespace(
        logits=mx.zeros((1, 1, 4)),
        hidden_states=[mx.zeros((1, 1, 4))],
        shared_kv_states={},
        cross_attention_states=None,
        encoder_outputs=None,
    )
    model.language_model.return_value = output

    embedding_output = MagicMock()
    embedding_output.inputs_embeds = mx.zeros((1, 5, 4))
    embedding_output.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding_output

    draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[]),
    )

    with (
        patch("mlx_vlm.speculative.drafters.validate_drafter_compatibility"),
        patch.object(generate_module.cache, "make_prompt_cache", return_value=[]),
        patch.object(generate_module, "make_logits_processors", return_value=[]),
        patch.object(
            generate_module, "make_sampler", return_value=lambda _: mx.array([0])
        ),
        patch.object(ar_module, "run_speculative_rounds", return_value=iter(())),
    ):
        gen = generate_module.generate_step(
            input_ids=mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
            model=model,
            pixel_values=None,
            mask=None,
            max_tokens=1,
            prefill_step_size=2,
            draft_model=draft_model,
            draft_kind="mtp",
        )
        list(gen)

    assert model.language_model.call_args_list[0].kwargs["n_to_process"] == 2
    assert model.language_model.call_args_list[1].kwargs["n_to_process"] == 2
    model.chunked_prefill_policy.assert_called_once()


def test_chunked_prefill_policy_defaults_conservative_for_speculation():
    model = SimpleNamespace(no_chunked_prefill=False)

    assert ar_module._chunked_prefill_enabled(model)
    assert not ar_module._chunked_prefill_enabled(
        model,
        draft_model=SimpleNamespace(config=SimpleNamespace(target_layer_ids=[])),
        draft_kind="mtp",
        prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
    )


def test_stream_generate_forwards_verbose_to_generate_step():
    captured = {}

    class FakeStoppingCriteria:
        def __call__(self, token):
            return False

    class FakeDetokenizer:
        def reset(self):
            self.segments = []

        def add_token(self, token, skip_special_token_ids=None):
            self.segments.append(str(token))

        @property
        def last_segment(self):
            return self.segments.pop(0) if self.segments else ""

        def finalize(self):
            pass

    def fake_generate_step(*args, **kwargs):
        captured["verbose"] = kwargs.get("verbose")
        yield 7, mx.zeros((4,))

    tokenizer = SimpleNamespace(stopping_criteria=FakeStoppingCriteria())
    processor = SimpleNamespace(tokenizer=tokenizer, detokenizer=FakeDetokenizer())
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="test", eos_token_id=[]),
        language_model=SimpleNamespace(),
    )

    with patch.object(dispatch_module, "generate_step", side_effect=fake_generate_step):
        list(
            dispatch_module.stream_generate(
                model=model,
                processor=processor,
                prompt="",
                input_ids=mx.array([[1]], dtype=mx.int32),
                pixel_values=None,
                mask=None,
                prompt_cache=[],
                max_tokens=1,
                verbose=True,
            )
        )

    assert captured["verbose"] is True


def test_normalize_resize_shape_expands_single_value():
    assert normalize_resize_shape([224]) == (224, 224)


def test_normalize_resize_shape_accepts_two_values():
    assert normalize_resize_shape((224, 448)) == (224, 448)


@pytest.mark.parametrize("value", [224, "22", [1.5], [True], [1, 2, 3]])
def test_normalize_resize_shape_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="resize_shape must contain 1 or 2 integers"):
        normalize_resize_shape(value)


def test_generate_cli_smoke(capsys):
    prompt = ["画像の前面に積み重なっているのはどのようなアイテムですか？"]

    args = Namespace(
        model="demo",
        output_modality="text",
        output=None,
        size="512x512",
        steps=4,
        seed=None,
        guidance=1.0,
        adapter_path=None,
        image=["image.png"],
        audio=None,
        video=None,
        fps=2.0,
        resize_shape=[224],
        prompt=prompt,
        system=None,
        max_tokens=12,
        temperature=0.7,
        repetition_penalty=None,
        repetition_context_size=20,
        presence_penalty=None,
        presence_context_size=20,
        frequency_penalty=None,
        frequency_context_size=20,
        chat=False,
        verbose=False,
        eos_tokens=None,
        max_kv_size=None,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=512,
        skip_special_tokens=False,
        force_download=False,
        revision="main",
        trust_remote_code=False,
        quantize_activations=False,
        processor_kwargs={},
        prefill_step_size=128,
        enable_thinking=False,
        thinking_budget=None,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        draft_model=None,
        draft_kind="dflash",
        draft_block_size=None,
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="demo"))
    processor = SimpleNamespace()

    with (
        patch.object(dispatch_module, "parse_arguments", return_value=args),
        patch.object(dispatch_module, "load", return_value=(model, processor)),
        patch.object(
            dispatch_module, "apply_chat_template", return_value="prompt"
        ) as mock_apply_chat_template,
        patch.object(
            dispatch_module,
            "generate",
            return_value=SimpleNamespace(text="done"),
        ) as mock_generate,
    ):
        dispatch_module.main()

    assert mock_apply_chat_template.call_args.args[2] == prompt
    assert mock_apply_chat_template.call_args.kwargs["num_images"] == 1
    assert mock_apply_chat_template.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["temperature"] == pytest.approx(0.7)
    assert mock_generate.call_args.kwargs["prefill_step_size"] == 128
    assert capsys.readouterr().out.strip() == "done"


def test_generate_image_cli_routes_before_vlm_load():
    args = Namespace(
        model="bonsai-ternary",
        output_modality="image",
        task="generate",
        output="out.png",
        size="512x512",
        steps=4,
        seed=7,
        guidance=1.0,
    )

    with (
        patch.object(dispatch_module, "parse_arguments", return_value=args),
        patch.object(dispatch_module, "run_image_generation_cli") as mock_run_image,
        patch.object(dispatch_module, "load") as mock_load,
    ):
        dispatch_module.main()

    mock_run_image.assert_called_once_with(args)
    mock_load.assert_not_called()


def test_generate_image_cli_edit_task_loads_edit_model_and_saves_output(tmp_path):
    output_path = tmp_path / "edited.png"
    args = Namespace(
        model="black-forest-labs/FLUX.2-klein-9b-kv",
        task="edit",
        image=["reference.png"],
        prompt=["add", "sunglasses"],
        output=str(output_path),
        size="256x512",
        steps=2,
        seed=7,
        guidance=1.0,
    )
    result = SimpleNamespace(
        path=output_path,
        seed=7,
        width=256,
        height=512,
        steps=2,
        variant="flux2-klein-9b-kv",
    )
    model = SimpleNamespace()

    with (
        patch.object(image_module, "load_image_model", return_value=model),
        patch.object(image_module, "generate_image", return_value=result) as mock_edit,
    ):
        image_module.run_image_generation_cli(args)

    edit_request = mock_edit.call_args.args[1]
    assert edit_request.prompt == "add sunglasses"
    assert edit_request.image_paths == ("reference.png",)
    assert edit_request.width == 256
    assert edit_request.height == 512
    assert mock_edit.call_args.kwargs["task"] == "edit"
    assert mock_edit.call_args.kwargs["output_path"] == output_path


def test_parse_arguments_defaults_thinking_tokens(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mlx_vlm.generate"])

    args = generate_module.parse_arguments()

    assert args.thinking_start_token == "<think>"
    assert args.thinking_end_token == "</think>"
    assert args.output_modality == "text"
    assert args.task == "generate"
    assert args.size is None


def test_cached_prefix_rope_failure_falls_back_to_cold(caplog):
    class BrokenRopeLanguageModel:
        def __init__(self):
            self._rope_deltas = mx.array([1])
            self._position_ids = mx.array([[0, 1, 2]])

        def get_rope_index(self, *args, **kwargs):
            raise ValueError("bad grid")

    language_model = BrokenRopeLanguageModel()
    model = SimpleNamespace(language_model=language_model)
    rope_deltas_before = language_model._rope_deltas
    position_ids_before = language_model._position_ids
    kwargs = {}

    with caplog.at_level(logging.WARNING, logger="mlx_vlm.generate"):
        ok = _prime_cached_prefix_rope_state(
            model,
            mx.array([[1, 2, 3]]),
            None,
            kwargs,
        )

    assert ok is False
    assert "rope_deltas" not in kwargs
    assert bool(mx.array_equal(language_model._rope_deltas, rope_deltas_before))
    assert bool(mx.array_equal(language_model._position_ids, position_ids_before))
    assert "falling back to cold prefill" in caplog.text


def test_batch_apc_extra_hash_uses_precomputed_image_hash():
    batch_generator = SimpleNamespace(apc_manager=object())

    got = BatchGenerator._apc_extra_hash(
        batch_generator,
        {"_apc_image_hash": 123, "_apc_tenant": "tenant-a"},
    )

    assert got == apc_module.tenant_scoped_hash("tenant-a", 123)


def test_cold_batch_left_pads_sequence_aligned_prompt_kwargs():
    class EmptyGenerationBatch:
        def __len__(self):
            return 0

    bg = object.__new__(BatchGenerator)
    bg._generation_batch = EmptyGenerationBatch()
    bg._prompt_batch = None
    bg._prompt_tokens_counter = 0
    bg._prompt_time_counter = 0
    bg._gen_tokens_counter = 0
    bg._steps_counter = 0
    bg.completion_batch_size = 4
    bg.prefill_batch_size = 4
    bg.prefill_step_size = 1
    bg.kv_bits = None
    bg.kv_group_size = 64
    bg.kv_quant_scheme = "affine"
    bg.apc_manager = None
    bg.apc_mode = None
    bg.model = SimpleNamespace()
    bg._wire_stack = None
    bg.compute_logprobs = False
    bg.top_logprobs_k = 0
    bg.sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    bg.tokenizer = SimpleNamespace(stopping_criteria=object())

    lengths = [2, 4, 3, 1]
    bg._unprocessed_sequences = [
        (
            i,
            list(range(length)),
            1,
            {
                "inputs_embeds": mx.ones((1, length, 3)) * (i + 1),
                "per_layer_inputs": mx.ones((1, length, 2, 5)) * (i + 1),
                "attention_mask": mx.ones((1, length), dtype=mx.int32),
                "pixel_values": mx.ones((1, 3, 2, 2)) * (i + 1),
                "keep_tensor": mx.array([[i + 1]], dtype=mx.int32),
                "rope_deltas": mx.array([[i + 10]], dtype=mx.int32),
                "_apc_tenant": "tenant",
            },
            [],
            None,
        )
        for i, length in enumerate(lengths)
    ]

    captured = {}

    def fake_prompt_batch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            total_prompt_tokens=sum(len(ids) for ids in kwargs["input_ids"]),
            needs_processing=lambda: True,
            prompt_step=lambda: 0,
        )

    with patch.object(generate_module, "PromptProcessingBatch", fake_prompt_batch):
        bg._next()

    prompt_kwargs = captured["prompt_kwargs"]
    assert captured["inputs_embeds"].shape == (4, 4, 3)
    assert prompt_kwargs["per_layer_inputs"].shape == (4, 4, 2, 5)
    assert prompt_kwargs["attention_mask"].shape == (4, 4)
    assert prompt_kwargs["pixel_values"].shape == (4, 3, 2, 2)
    assert prompt_kwargs["keep_tensor"].shape == (4, 1)
    assert prompt_kwargs["rope_deltas"].shape == (4, 1)
    assert "_apc_tenant" not in prompt_kwargs
    assert prompt_kwargs["per_layer_inputs"][0, :, 0, 0].tolist() == [0, 0, 1, 1]
    assert prompt_kwargs["per_layer_inputs"][3, :, 0, 0].tolist() == [0, 0, 0, 4]


def test_mixed_apc_batch_strips_private_kwargs_before_prefill():
    bg = object.__new__(BatchGenerator)
    bg.apc_manager = object()
    bg.model = SimpleNamespace(layers=[object()])
    bg.prefill_step_size = None
    bg.kv_bits = None
    bg.kv_group_size = 64
    bg.kv_quant_scheme = "affine"
    bg._wire_stack = None

    captured = {}

    def fake_prompt_batch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    sequences = [
        (
            1,
            list(range(8)),
            1,
            {
                "inputs_embeds": mx.ones((1, 8, 4)),
                "keep_tensor": mx.ones((1, 1)),
                "_apc_tenant": "tenant-a",
                "_apc_image_hash": 123,
            },
            [],
            None,
        ),
        (
            2,
            list(range(6)),
            1,
            {
                "inputs_embeds": mx.ones((1, 6, 4)),
                "keep_tensor": mx.zeros((1, 1)),
                "_apc_tenant": "tenant-b",
                "_apc_image_hash": 456,
            },
            [],
            None,
        ),
    ]
    picks = [
        {
            "matched_blocks": [],
            "prefix_len": 4,
            "extra_hash": 7,
            "full_input_ids": list(range(8)),
        },
        None,
    ]

    with (
        patch.object(BatchGenerator, "_apc_pick_for", side_effect=picks),
        patch.object(
            ar_module._apc,
            "make_warm_batch_kv_cache_multi",
            return_value=([], 4),
        ),
        patch.object(generate_module, "PromptProcessingBatch", fake_prompt_batch),
    ):
        batch = bg._build_mixed_prompt_batch(sequences)

    assert batch is not None
    assert "_apc_tenant" not in captured["prompt_kwargs"]
    assert "_apc_image_hash" not in captured["prompt_kwargs"]
    assert captured["prompt_kwargs"]["keep_tensor"].shape == (2, 1)


def test_apc_pick_rejects_image_tokens_and_releases_blocks():
    block_size = 4
    image_token_id = 99
    token_ids = [image_token_id, 1, 2, 3, 4]
    manager = apc_module.APCManager(num_blocks=4, block_size=block_size)
    layer_keys = [mx.ones((1, 1, block_size, 2))]
    layer_values = [mx.ones((1, 1, block_size, 2)) * 2]
    stored = manager.store_kv_blocks(
        token_ids[:block_size],
        layer_keys,
        layer_values,
    )
    manager.release(stored)

    bg = object.__new__(BatchGenerator)
    bg.apc_manager = manager
    bg.model = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id))
    bg._wire_stack = None

    pick = bg._apc_pick_for((1, token_ids, 1, {}, [], None))

    assert pick is None
    assert all(block.ref_cnt == 0 for block in stored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
