"""Generation, sampling, stopping criteria, structured logits, and thinking state."""

from __future__ import annotations

import contextlib
import gc
import importlib
import logging
import sys
import types
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import apc as apc_module
from mlx_vlm import sample_utils as sampling
from mlx_vlm import structured
from mlx_vlm.generate import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    GenerationBatch,
    PromptProcessingBatch,
    SpeculativeGenerationBatch,
    _left_pad_prompts,
    _prime_cached_prefix_rope_state,
)
from mlx_vlm.generate import ar as ar_module
from mlx_vlm.generate import dispatch as dispatch_module
from mlx_vlm.generate import normalize_resize_shape
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import (
    BatchKVCache,
    BufferedRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx_vlm.structured import ThinkingAwareLogitsProcessor
from mlx_vlm.utils import StoppingCriteria, ThinkingBudgetCriteria, load_processor

# Generation, sampling, stopping criteria, structured logits, and thinking state.

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


def _input_shape(input_ids):
    return input_ids.shape if input_ids.ndim > 1 else (1, input_ids.shape[0])


class MockLanguageModel:
    """Mock language model for testing batch generation."""

    def __init__(self, vocab_size=32000, hidden_size=768):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers = [MagicMock() for _ in range(4)]

    def __call__(self, input_ids, cache=None, **kwargs):
        return MagicMock(
            logits=mx.random.normal((*_input_shape(input_ids), self.vocab_size))
        )


class MockModel:
    """Mock VLM model for testing."""

    def __init__(self):
        self.config = MockConfig()
        self.language_model = MockLanguageModel()

    def __call__(self, input_ids, pixel_values=None, cache=None, mask=None, **kwargs):
        return MagicMock(
            logits=mx.random.normal((*_input_shape(input_ids), 32000)),
            cross_attention_states=None,
            encoder_outputs=None,
        )

    def get_input_embeddings(self, input_ids, pixel_values, **kwargs):
        return mx.random.normal((*_input_shape(input_ids), 768))

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


def test_batch_generator_apc_media_token_ids_handles_text_only_model(mock_processor):
    model = SimpleNamespace(
        language_model=MockLanguageModel(), make_cache=lambda: [KVCache()]
    )
    generator = ar_module.BatchGenerator(
        model, mock_processor, apc_manager=apc_module.APCManager(num_blocks=1)
    )

    assert generator._apc_media_token_ids() == set()


class TestGenerationBatch:
    """Tests for GenerationBatch class."""

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

    def test_filter_releases_pending_arrays_without_cyclic_gc(self):
        batch = self._mrope_batch([0], [[0]])
        pending_fields = (
            "_current_tokens",
            "_current_lps",
            "_next_tokens",
            "_next_lps",
            "_next_top_idx",
            "_next_top_lp",
            "_rope_deltas",
        )
        for field in pending_fields:
            setattr(batch, field, mx.ones((1, 2)))
        batch.prompt_cache = [
            SimpleNamespace(state=(mx.ones((1, 2, 3, 4)), [mx.zeros((1, 2, 3, 4))]))
        ]
        refs = [weakref.ref(getattr(batch, field)) for field in pending_fields]
        refs.extend(
            (
                weakref.ref(batch.prompt_cache[0].state[0]),
                weakref.ref(batch.prompt_cache[0].state[1][0]),
            )
        )

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            batch.filter([])
            assert all(ref() is None for ref in refs)
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()

    def test_eval_pending_state_materializes_nested_arrays(self, monkeypatch):
        batch = self._mrope_batch([0], [[0]])
        batch._next_tokens = mx.array([5]) + 1
        batch.prompt_cache = [
            SimpleNamespace(state=(mx.array([1]) + 2, [None, mx.array([3]) + 4])),
            SimpleNamespace(),
        ]
        expected = (
            batch._next_tokens,
            batch._rope_deltas,
            batch.prompt_cache[0].state[0],
            batch.prompt_cache[0].state[1][1],
        )
        evaluated = []
        original_eval = mx.eval

        def record_eval(*arrays):
            evaluated.extend(id(array) for array in arrays)
            original_eval(*arrays)

        monkeypatch.setattr(mx, "eval", record_eval)
        batch._eval_pending_state()
        assert set(evaluated) == {id(array) for array in expected}
        assert len(evaluated) == len(expected)
        assert [array.tolist() for array in expected] == [[6], [[0]], [3], [7]]

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

    def test_capture_rope_deltas_prefers_prompt_kwargs(self):
        from mlx_vlm.generate import PromptProcessingBatch

        captured = PromptProcessingBatch._capture_rope_deltas_from_prompt_kwargs(
            {"rope_deltas": mx.array([[5], [7]], dtype=mx.int32)},
            SimpleNamespace(_rope_deltas=mx.array([[99], [99]], dtype=mx.int32)),
            2,
        )
        assert captured.tolist() == [[5], [7]]


# ============================================================================
# Tests for Helper Functions
# ============================================================================


class TestLeftPadPrompts:
    """Tests for _left_pad_prompts function."""

    def test_single_prompt(self):
        prompts = [[1, 2, 3, 4, 5]]
        padded = _left_pad_prompts(prompts)

        assert padded.shape == (1, 5)
        assert mx.array_equal(padded[0], mx.array([1, 2, 3, 4, 5]))


# ============================================================================
# Tests for BatchGenerator Class
# ============================================================================


class FixedLogitModel:
    def __call__(self, input_ids, cache=None, **kwargs):
        scores = mx.array([0.0, 10.0, 0.0, 0.0])
        return MagicMock(logits=mx.broadcast_to(scores, (*input_ids.shape, 4)))


def _generation_batch(model, inputs=(5, 6), uids=None, **options):
    options = (
        dict(
            prompt_cache=[],
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            stop_criteria=lambda token: False,
            max_tokens=[2] * len(inputs),
        )
        | options
    )
    return GenerationBatch(
        model=model,
        inputs=mx.array(inputs, mx.int32),
        uids=list(range(len(inputs))) if uids is None else uids,
        **options,
    )


class TestBatchGenerator:
    """Tests for BatchGenerator class."""

    @pytest.mark.parametrize("phased", [False, True])
    @pytest.mark.parametrize("budget", [1, 2, 7, None])
    @pytest.mark.parametrize("capacity", [1, 3])
    def test_prefill_budget_and_partial_admission(
        self, mock_processor, budget, capacity, phased
    ):
        import mlx.nn as nn

        calls = []

        class Model(nn.Module):
            def make_cache(self):
                return [KVCache()]

            def __call__(self, ids, cache=None, inputs_embeds=None, **kwargs):
                calls.append((inputs_embeds is not None, ids.shape))
                kv = mx.zeros((ids.shape[0], 1, ids.shape[1], 4))
                cache[0].update_and_fetch(kv, kv)
                return SimpleNamespace(
                    logits=mx.broadcast_to(mx.array([0.0, 10.0, 0.0]), (*ids.shape, 3))
                )

        gen = BatchGenerator(
            Model(),
            mock_processor,
            completion_batch_size=capacity,
            prefill_batch_size=8,
            prefill_step_size=budget,
            compute_logprobs=False,
        )

        def step():
            if not phased:
                return gen.next()
            responses = gen.decode_step()
            return gen.prefill_step(), responses

        def insert(prompts, max_tokens):
            return gen.insert(
                prompts,
                max_tokens=max_tokens,
                prompt_kwargs=[
                    {"inputs_embeds": mx.zeros((1, len(ids), 4))} for ids in prompts
                ],
            )

        (first,) = insert([[4, 5]], 32)
        for _ in range(3):
            step()
            if len(gen._generation_batch):
                break
        assert gen._generation_batch.uids == [first]

        # Requests arriving during decoding can use the remaining slots even
        # when fewer than prefill_batch_size are available.
        late = insert([[6] * 7, [7] * 5, [8] * 9], 2)
        calls.clear()
        _, responses = step()
        assert [r.uid for r in responses] == [first]
        assert calls[0] == (False, (1, 1))
        if capacity > 1:
            assert any(is_prompt for is_prompt, _ in calls)

        finished = {r.uid for r in responses if r.finish_reason}
        for _ in range(120):
            if not gen.has_work:
                break
            _, responses = step()
            finished.update(r.uid for r in responses if r.finish_reason)
            assert len(gen._generation_batch) + len(gen._prompt_batch or []) <= capacity

        assert not gen.has_work
        assert finished == {first, *late}
        assert gen.stats().prompt_tokens == 23
        if budget is not None:
            assert all(b * n <= budget for is_prompt, (b, n) in calls if is_prompt)
        if capacity > 1 and budget != 1:
            assert any(is_prompt and b > 1 for is_prompt, (b, _) in calls)
        gen.close()

    def test_insert_with_max_tokens(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model, processor=mock_processor, max_tokens=50
        )

        prompts = [[1, 2, 3], [4, 5]]
        max_tokens = [100, 200]
        uids = gen.insert(prompts, max_tokens=max_tokens)

        assert len(uids) == 2
        # Prompts are sorted by length, so check the unprocessed prompts
        assert len(gen.unprocessed_prompts) == 2

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
        uids = gen.insert([prompt], prompt_kwargs=[{"inputs_embeds": inputs_embeds}])
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

    def test_chunked_prefill_stats_count_each_prompt_token_once(
        self, mock_model, mock_processor
    ):
        gen = BatchGenerator(
            model=mock_model.language_model,
            processor=mock_processor,
            prefill_batch_size=1,
            completion_batch_size=1,
            prefill_step_size=2,
        )
        prompt_tokens = 5
        gen._prompt_batch = SimpleNamespace(
            needs_processing=lambda: True, prompt_step=lambda: 2
        )
        gen._prompt_tokens_counter = prompt_tokens

        gen.next()

        assert gen.stats().prompt_tokens == prompt_tokens

    def test_generation_batch_applies_per_sequence_logits_processors(self):
        seen_contexts = []

        def force_token_2(tokens, logits):
            seen_contexts.append(tokens.tolist())
            token_scores = mx.array([-1e9, -1e9, 0.0, -1e9])
            return mx.broadcast_to(token_scores, logits.shape)

        batch = _generation_batch(
            model=FixedLogitModel(),
            token_context=[mx.array([10]), mx.array([20])],
            logits_processors=[[force_token_2], [force_token_2]],
        )

        first = batch.next()
        assert [r.token for r in first] == [5, 6]
        assert seen_contexts == [[10, 5], [20, 6]]

        second = batch.next()
        assert [r.token for r in second] == [2, 2]

    def test_generation_batch_thinking_budget_criteria_can_force_next_token(self):
        class ForceAfterFirst:
            def __init__(self):
                self.forced_token_id = None

            def __call__(self, token):
                self.forced_token_id = 3 if token == 5 else None

            def pop_forced_token_id(self):
                forced = self.forced_token_id
                self.forced_token_id = None
                return forced

        batch = _generation_batch(
            model=FixedLogitModel(),
            inputs=(5,),
            thinking_budget_criteria=[ForceAfterFirst()],
        )

        first = batch.next()
        assert [r.token for r in first] == [5]

        second = batch.next()
        assert [r.token for r in second] == [3]

    def test_generation_batch_uses_fused_greedy_decode_without_logprobs(self):
        class FastArgmaxModel:
            def __init__(self):
                self.calls = []

            def fused_greedy_decode(self, input_ids, cache=None, **kwargs):
                self.calls.append(kwargs)
                assert cache == []
                return mx.full(
                    (input_ids.shape[0], input_ids.shape[1]), 7, dtype=mx.int32
                )

        model = FastArgmaxModel()
        batch = _generation_batch(model=model, greedy_sampling=True)
        batch.compute_logprobs = False

        first = batch.next()
        assert [r.token for r in first] == [5, 6]
        assert batch._next_tokens.tolist() == [7, 7]
        assert model.calls == [{}]

    def test_generation_batch_ignores_speculative_argmax_without_fused_decode(self):
        class FallbackArgmaxModel:
            def __init__(self):
                self.calls = []

            def __call__(self, input_ids, cache=None, **kwargs):
                del cache
                self.calls.append(kwargs)
                logits = mx.broadcast_to(
                    mx.array([0.0, 1.0, 4.0, 2.0]),
                    (input_ids.shape[0], input_ids.shape[1], 4),
                )
                return SimpleNamespace(logits=logits)

            def speculative_argmax_from_hidden(self, hidden):
                raise AssertionError("fallback argmax must not select the fused path")

        model = FallbackArgmaxModel()
        batch = _generation_batch(model=model, inputs=(5,), greedy_sampling=True)
        batch.compute_logprobs = False

        first = batch.next()
        assert [r.token for r in first] == [5]
        assert batch._next_tokens.tolist() == [2]
        assert model.calls == [{}]

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

    def test_generation_batch_extend_expands_compact_processor_state(self):
        sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        stop_criteria = lambda token: False

        def make_batch(uid, processor=None):
            return _generation_batch(
                model=object(),
                uids=[uid],
                inputs=(uid + 1,),
                sampler=sampler,
                stop_criteria=stop_criteria,
                token_context=[[30]] if processor is not None else None,
                logits_processors=[[processor]] if processor is not None else None,
            )

        first_plain = make_batch(0)
        second_plain = make_batch(1)
        structured_processor = lambda tokens, logits: logits
        structured = make_batch(2, structured_processor)

        first_plain.extend(second_plain)
        assert first_plain.logits_processors == []

        first_plain.extend(structured)

        assert first_plain.uids == [0, 1, 2]
        assert first_plain.token_context == [[], [], [30]]
        assert first_plain.logits_processors == [None, None, [structured_processor]]

    def test_generation_batch_extend_promotes_singleton_kv_cache(self):
        def make_kv_cache(value):
            c = KVCache()
            keys = mx.full((1, 2, 3, 4), value, dtype=mx.float32)
            values = mx.full((1, 2, 3, 4), value + 1, dtype=mx.float32)
            c.update_and_fetch(keys, values)
            return c

        sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
        stop_criteria = lambda token: False
        first = _generation_batch(
            model=MagicMock(),
            inputs=(5,),
            prompt_cache=[make_kv_cache(1.0)],
            sampler=sampler,
            stop_criteria=stop_criteria,
        )
        second = _generation_batch(
            model=MagicMock(),
            uids=[1],
            inputs=(6,),
            prompt_cache=[make_kv_cache(3.0)],
            sampler=sampler,
            stop_criteria=stop_criteria,
        )

        first.extend(second)

        assert isinstance(first.prompt_cache[0], BatchKVCache)
        assert first.prompt_cache[0].left_padding.tolist() == [0, 0]
        assert first.prompt_cache[0].keys.shape[0] == 2
        assert first._next_tokens.tolist() == [5, 6]

    def test_remove_from_unprocessed(self, mock_model, mock_processor):
        gen = BatchGenerator(
            model=mock_model.language_model, processor=mock_processor, max_tokens=50
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
            model=mock_model.language_model, processor=mock_processor, max_tokens=50
        )
        gen.insert([[1, 2, 3]])
        assert gen.remove(9999) is False

    def test_remove_cancels_image_prefill_and_releases_cache(
        self, mock_model, mock_processor
    ):
        gen = BatchGenerator(
            model=mock_model.language_model, processor=mock_processor, max_tokens=50
        )
        prompt_batch = SimpleNamespace(
            uids=[7],
            prompt_cache=[MagicMock()],
            input_ids=mx.array([[1, mock_model.config.image_token_index, 2]]),
        )
        gen._prompt_batch = prompt_batch

        assert gen.remove(7) is True
        assert prompt_batch.uids == []
        assert prompt_batch.prompt_cache == []
        assert gen._prompt_batch is None


# ============================================================================
# Tests for batch_generate function
# ============================================================================


@pytest.mark.parametrize(
    "case", ["text", "images", "no-sizes", "image-string", "verbose"]
)
def test_batch_generate_media(case, mock_model, mock_processor, capsys):
    from PIL import Image

    from mlx_vlm.generate import batch_generate

    prompts, images, options = ["Describe this"], ["test.jpg"], {}
    count = 2 if case in ("text", "images") else 1
    texts = ["Response 1", "Response 2"] if count == 2 else ["Response"]
    stats = BatchStats(prompt_tokens=20, generation_tokens=10)
    if case == "text":
        prompts, images, options = ["Hello", "World"], None, dict(max_tokens=50)
    elif case == "images":
        prompts = ["Describe image 1", "Describe image 2"]
        images = ["path/to/img1.jpg", "path/to/img2.jpg"]
        options = dict(max_tokens=50)
        stats = BatchStats(prompt_tokens=40, generation_tokens=20)
    elif case == "no-sizes":
        options = dict(track_image_sizes=False)
    elif case == "image-string":
        images, stats = "single_image.jpg", BatchStats()
    else:
        options = dict(verbose=True)
        stats = BatchStats(
            prompt_tokens=100,
            prompt_time=0.1,
            generation_tokens=50,
            generation_time=0.2,
        )
    size = (512, 384) if case == "no-sizes" else (224, 224)
    with (
        patch.object(ar_module, "_generate_batch", return_value=(texts, stats)) as run,
        patch(
            "mlx_vlm.utils.process_image",
            side_effect=[
                MagicMock(spec=Image.Image, height=size[0], width=size[1])
                for _ in range(count)
            ],
        ) as process,
    ):
        response = batch_generate(
            mock_model, mock_processor, images=images, prompts=prompts, **options
        )
    assert isinstance(response, BatchResponse)
    assert response.texts == texts
    run.assert_called_once()
    if case == "no-sizes":
        assert response.image_sizes is None
    if case == "image-string":
        process.assert_called_once()
    if case == "verbose":
        assert "[batch_generate]" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["audios", "videos"])
def test_batch_generate_routes_audio_and_video(kind, mock_model, mock_processor):
    media = [f"{kind}-0", f"{kind}-1"]
    stats = BatchStats(prompt_tokens=20, generation_tokens=10)
    if kind == "audios":
        mock_processor.supports_multiple_audio = True
    with patch.object(
        ar_module, "_generate_batch", return_value=(["first", "second"], stats)
    ) as run:
        response = ar_module.batch_generate(
            mock_model,
            mock_processor,
            prompts=["one", "two"],
            **{kind: media},
        )

    assert response.texts == ["first", "second"]
    assert run.call_args.kwargs[kind] == media


def test_batch_generate_combines_media_and_text(mock_model, mock_processor):
    stats = BatchStats()
    with patch.object(
        ar_module,
        "_generate_batch",
        return_value=(["media", "text"], stats),
    ) as run:
        response = ar_module.batch_generate(
            mock_model,
            mock_processor,
            audios=["audio"],
            videos=["video"],
            prompts=["media prompt", "text prompt"],
            max_tokens=[20, 10],
        )

    assert response.texts == ["media", "text"]
    run.assert_called_once()
    assert run.call_args.kwargs["audios"] == ["audio"]
    assert run.call_args.kwargs["videos"] == ["video"]
    assert run.call_args.kwargs["max_tokens"] == [20, 10]


def test_batch_generate_rejects_more_media_than_prompts(mock_model, mock_processor):
    with pytest.raises(ValueError, match="3 audios for 2 prompts"):
        ar_module.batch_generate(
            mock_model,
            mock_processor,
            prompts=["one", "two"],
            audios=["a", "b", "c"],
        )


def test_batch_generate_rejects_unsupported_audio_batch(mock_model, mock_processor):
    with pytest.raises(ValueError, match="does not support batched audio"):
        ar_module.batch_generate(
            mock_model,
            mock_processor,
            prompts=["one", "two"],
            audios=["a", "b"],
        )


class TestBatchGenerate:
    """Tests for the batch_generate function."""

    def test_video_sampling_options_are_row_safe_and_not_generation_kwargs(
        self, mock_model, mock_processor
    ):
        class _StopGenerator(Exception):
            pass

        template_kwargs = []

        def apply_template(processor, config, prompt, **kwargs):
            template_kwargs.append(kwargs)
            return prompt

        def prepare(processor, **kwargs):
            assert kwargs["fps"] == [1]
            assert kwargs["nframes"] == 8
            assert kwargs["max_frames"] == 16
            return {
                "input_ids": mx.array([[1, 2], [3, 4]]),
                "attention_mask": mx.ones((2, 2)),
            }

        def make_generator(*args, **kwargs):
            assert "fps" not in kwargs
            assert "nframes" not in kwargs
            assert "max_frames" not in kwargs
            raise _StopGenerator

        embedding_output = InputEmbeddingsFeatures(inputs_embeds=mx.zeros((2, 2, 4)))
        with (
            patch.object(ar_module, "apply_chat_template", side_effect=apply_template),
            patch.object(ar_module, "prepare_inputs", side_effect=prepare),
            patch.object(
                mock_model, "get_input_embeddings", return_value=embedding_output
            ),
            patch.object(ar_module, "BatchGenerator", side_effect=make_generator),
            pytest.raises(_StopGenerator),
        ):
            ar_module._generate_batch(
                mock_model,
                mock_processor,
                prompts=["video", "text"],
                videos=["clip.mp4"],
                fps=[1],
                nframes=8,
                max_frames=16,
            )

        assert template_kwargs[0]["video"] == "clip.mp4"
        assert template_kwargs[0]["fps"] == 1
        assert template_kwargs[1]["video"] is None
        assert template_kwargs[1]["fps"] == 1

    def test_generate_batch_passes_mask_and_split_prompt_kwargs_to_generator(
        self, mock_model, mock_processor
    ):
        """BatchGenerator receives the dense rows and their padding metadata."""

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
            [[0, 0, 11, 12, 13], [21, 22, 23, 24, 25], [0, 31, 32, 33, 34]],
            dtype=mx.int32,
        )
        attention_mask = mx.array(
            [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=mx.int32
        )
        prepared_attention_mask = attention_mask
        inputs_embeds = mx.arange(
            batch_size * seq_len * hidden_size, dtype=mx.float32
        ).reshape(batch_size, seq_len, hidden_size)
        position_ids = mx.arange(batch_size * seq_len, dtype=mx.int32).reshape(
            batch_size, seq_len
        )
        embedding_output = _EmbeddingOutput(inputs_embeds, position_ids)

        def fake_insert(
            self,
            prompts,
            max_tokens,
            prompt_kwargs=None,
            logits_processors=None,
            attention_mask=None,
        ):
            assert mx.array_equal(prompts, input_ids)
            assert mx.array_equal(attention_mask, prepared_attention_mask)
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
                side_effect=lambda processor, config, prompt, **kwargs: prompt,
            ),
            patch.object(
                ar_module,
                "prepare_inputs",
                return_value={"input_ids": input_ids, "attention_mask": attention_mask},
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


# ============================================================================
# Edge Cases
# ============================================================================


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

    def _make_criteria(self, enable_thinking=True, prompt_preopens_thinking=True):
        return ThinkingBudgetCriteria(
            tokenizer=FakeTokenizer(),
            thinking_budget=5,
            thinking_end_token="</think>",
            thinking_start_token="<think>",
            enable_thinking=enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
        )

    def test_pop_forced_token_id_safe_after_start_delimiter(self):
        """Regression: the start-token early return in __call__ does not set
        forced_token_id, so pop_forced_token_id must remain safe afterwards."""
        criteria = self._make_criteria()
        # First generated token is the start delimiter -> early return, no force.
        assert criteria(99) is None
        assert criteria.pop_forced_token_id() is None

    def test_pop_forced_token_id_safe_after_end_delimiter(self):
        """Regression: the end-token early return in __call__ does not set
        forced_token_id, so pop_forced_token_id must remain safe afterwards."""
        criteria = self._make_criteria()
        # End delimiter resets thinking state and returns None without forcing.
        assert criteria(100) is None
        assert criteria.pop_forced_token_id() is None

    def test_pop_forced_token_id_consumes_pending_token_id(self):
        criteria = self._make_criteria()
        for i in range(5):
            assert criteria(50 + i) is None

        assert criteria(60) == 10
        assert criteria.pop_forced_token_id() == 10
        assert criteria.pop_forced_token_id() is None


@pytest.mark.parametrize("reused_prefix", [0, 1, 2])
def test_stream_generate_stores_checkpoint_only_before_decode(reused_prefix):
    class FakeStoppingCriteria:
        def __call__(self, token):
            return False

    class FakeDetokenizer:
        last_segment = ""

        def reset(self):
            pass

        def add_token(self, token, skip_special_token_ids=None):
            pass

        def finalize(self):
            pass

    coordinator = MagicMock()
    coordinator.enabled = True
    coordinator.is_checkpoint = True
    coordinator.lookup.return_value = (
        {"prefix_len": reused_prefix, "warm_cache": []} if reused_prefix else None
    )
    coordinator.materialize_single.return_value = []
    coordinator.checkpoint_lengths.return_value = [2, 3]

    def fake_generate_step(*args, **kwargs):
        coordinator.prepare_prefill.assert_called_once_with(4)
        assert args[0].shape[1] == 4 - reused_prefix
        for n in kwargs["prompt_cache_checkpoint_lengths"]:
            kwargs["prompt_cache_checkpoint"](n, kwargs["prompt_cache"])
        yield 7, mx.zeros((4,))

    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(stopping_criteria=FakeStoppingCriteria()),
        detokenizer=FakeDetokenizer(),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="test", eos_token_id=[]),
        language_model=SimpleNamespace(),
    )
    prompt_cache = []

    with (
        patch.object(dispatch_module._apc, "APCCoordinator", return_value=coordinator),
        patch.object(dispatch_module._apc, "semantic_extra_hash", return_value=0),
        patch.object(
            dispatch_module, "wired_limit", return_value=contextlib.nullcontext()
        ),
        patch.object(dispatch_module, "generate_step", side_effect=fake_generate_step),
    ):
        list(
            dispatch_module.stream_generate(
                model=model,
                processor=processor,
                prompt="",
                input_ids=mx.array([[1, 2, 3, 4]], dtype=mx.int32),
                pixel_values=None,
                mask=None,
                prompt_cache=prompt_cache,
                apc_manager=MagicMock(),
                max_tokens=1,
            )
        )

    calls = coordinator.store_checkpoint.call_args_list
    assert [call.args[0] for call in calls] == [
        [1, 2, 3, 4][:n] for n in [2, 3] if n > reused_prefix
    ]
    assert all(call.args[1] == prompt_cache for call in calls)
    assert all(call.kwargs == {"extra_hash": 0} for call in calls)


@pytest.mark.parametrize("value", [224, "22", [1.5], [True], [1, 2, 3]])
def test_normalize_resize_shape_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="resize_shape must contain 1 or 2 integers"):
        normalize_resize_shape(value)


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
        ok = _prime_cached_prefix_rope_state(model, mx.array([[1, 2, 3]]), None, kwargs)

    assert ok is False
    assert "position_ids" not in kwargs
    assert "rope_deltas" not in kwargs
    assert bool(mx.array_equal(language_model._rope_deltas, rope_deltas_before))
    assert bool(mx.array_equal(language_model._position_ids, position_ids_before))
    assert "falling back to cold prefill" in caplog.text


def test_cached_prefix_rope_forwards_full_prompt_metadata():
    position_ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    rope_deltas = mx.array([[7]], dtype=mx.int32)

    class RopeLanguageModel:
        _position_ids = None
        _rope_deltas = None

        @staticmethod
        def get_rope_index(*args, **kwargs):
            return position_ids, rope_deltas

    language_model = RopeLanguageModel()
    kwargs = {}

    ok = _prime_cached_prefix_rope_state(
        SimpleNamespace(language_model=language_model),
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        None,
        kwargs,
    )

    assert ok is True
    assert bool(mx.array_equal(kwargs["position_ids"], position_ids))
    assert bool(mx.array_equal(kwargs["rope_deltas"], rope_deltas))
    assert bool(mx.array_equal(language_model._position_ids, position_ids))
    assert bool(mx.array_equal(language_model._rope_deltas, rope_deltas))


class TestPrefixCacheReuseTrim:
    """Prompt-cache prefix reuse must trim each cache through its own trim()
    contract. A raw ``keys[..., :prefix_len, :]`` slice corrupts rotating
    (sliding-window) ring buffers -- silent wrong output, or a shape crash once
    speculative decoding wraps them (mlx-vlm issue #1715)."""

    @staticmethod
    def _fill(cache, n, marker=False, heads=1, dim=4):
        for i in range(n):
            v = (
                mx.full((1, heads, 1, dim), float(i))
                if marker
                else mx.zeros((1, heads, 1, dim))
            )
            cache.update_and_fetch(v, v)
        return cache

    def test_unwrapped_rotating_trims_and_stays_usable(self):
        c = self._fill(RotatingKVCache(max_size=512), 100)
        assert c.offset == 100  # window has not wrapped
        n_drop = dispatch_module._prefix_cache_trim_amount([c], 40)
        assert n_drop == 60
        c.trim(n_drop)
        assert c.offset == 40 and c._idx == 40
        c.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
        assert c.offset == 41

    def test_mixed_flat_and_wrapped_rotating_is_not_reusable(self):
        flat = self._fill(KVCache(), 20)
        wrapped = self._fill(RotatingKVCache(max_size=8), 20)
        assert dispatch_module._prefix_cache_trim_amount([flat, wrapped], 3) is None

    def test_wrapped_buffered_rotating_declined_and_survives_next_step(self):
        # BufferedRotatingKVCache is what speculative decoding installs; the old
        # raw slice desynced its ring index and crashed on the next update. When
        # reuse is declined the cache stays intact, so generation continues.
        c = BufferedRotatingKVCache.from_cache(
            self._fill(RotatingKVCache(max_size=8), 20), buffer_size=16
        )
        assert dispatch_module._prefix_cache_trim_amount([c], 2) is None
        c.update_and_fetch(mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))


@pytest.mark.parametrize("honors_hint", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("right_padded", [False, True])
@pytest.mark.parametrize(
    "prefill_step_size,chunks", [(None, 0), (2, 0), (2, 1), (2, None)]
)
def test_prompt_processing_requests_only_required_trailing_logits(
    right_padded, batch_size, prefill_step_size, chunks, honors_hint
):
    import mlx.nn as nn

    calls = []

    class Model(nn.Module):
        def make_cache(self):
            return [KVCache()]

        def __call__(self, input_ids, cache=None, **kwargs):
            calls.append((input_ids.shape[1], kwargs))
            kv = mx.zeros((input_ids.shape[0], 1, input_ids.shape[1], 4))
            cache[0].update_and_fetch(kv, kv)
            logits = (input_ids[..., None] == mx.arange(16)).astype(mx.float32)
            if honors_hint:
                logits = logits[:, -kwargs.get("logits_to_keep", input_ids.shape[1]) :]
            return SimpleNamespace(logits=logits)

    input_ids = [[1, 2, 3, 4, 5], [6, 7, 8]][:batch_size]
    right_padding = [0, 2][:batch_size] if right_padded else None
    batch = PromptProcessingBatch(
        model=Model(),
        uids=list(range(batch_size)),
        input_ids=input_ids,
        max_tokens=[1] * batch_size,
        inputs_embeds=mx.zeros((batch_size, 5, 4)),
        prompt_kwargs={},
        prefill_step_size=prefill_step_size,
        right_pad_per_row=right_padding,
        greedy_sampling=True,
    )

    if chunks is None:
        while batch.needs_processing():
            assert batch.prompt_step() > 0
    else:
        for _ in range(chunks):
            assert batch.prompt_step() > 0

    expected_input_width = 1 if chunks is None else 5 - 2 * chunks
    assert batch._input_ids.shape[1] == expected_input_width

    gen_batch = batch.generate(
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        stop_criteria=[lambda _: False] * batch_size,
        compute_logprobs=False,
    )

    final_input_width, final_kwargs = calls[-1]
    assert final_input_width == expected_input_width
    expected_keep = 1 if not right_padded or batch_size == 1 or chunks is None else 3
    assert final_kwargs["logits_to_keep"] == expected_keep
    assert gen_batch._next_tokens.tolist() == [row[-1] for row in input_ids]


def test_precomputed_semantic_hash_reuses_actual_growing_apc_prefix():
    manager = apc_module.APCManager(num_blocks=4, block_size=4)
    manager.exact_cache_guard_tokens = 1
    semantic_hash = 7088136067003016882
    short_tokens = list(range(8))
    extended_tokens = [*short_tokens, 8, 9]
    layer_keys = [mx.ones((1, 1, len(short_tokens), 2))]
    layer_values = [mx.ones((1, 1, len(short_tokens), 2)) * 2]

    stored = manager.store_kv_blocks(
        short_tokens, layer_keys, layer_values, extra_hash=semantic_hash
    )
    manager.release(stored)

    batch_generator = object.__new__(BatchGenerator)
    batch_generator.apc_manager = manager
    batch_generator.apc_mode = "block"
    batch_generator.model = SimpleNamespace(config=SimpleNamespace())
    batch_generator._wire_stack = None

    pick = batch_generator._apc_pick_for(
        (1, extended_tokens, 1, {"_apc_semantic_hash": semantic_hash}, [], None)
    )

    assert pick is not None
    assert pick["prefix_len"] == len(short_tokens)
    assert pick["extra_hash"] == semantic_hash
    manager.release(pick["matched_blocks"])


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
    bg.prefill_step_size = 4
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
                "position_ids": mx.ones((3, 1, length), dtype=mx.int32) * (i + 1),
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
    assert prompt_kwargs["position_ids"].shape == (3, 4, 4)
    assert prompt_kwargs["pixel_values"].shape == (4, 3, 2, 2)
    assert prompt_kwargs["keep_tensor"].shape == (4, 1)
    assert prompt_kwargs["rope_deltas"].shape == (4, 1)
    assert "_apc_tenant" not in prompt_kwargs
    assert prompt_kwargs["per_layer_inputs"][0, :, 0, 0].tolist() == [0, 0, 1, 1]
    assert prompt_kwargs["per_layer_inputs"][3, :, 0, 0].tolist() == [0, 0, 0, 4]
    assert prompt_kwargs["position_ids"][0, 0].tolist() == [0, 0, 1, 1]
    assert prompt_kwargs["position_ids"][0, 3].tolist() == [0, 0, 0, 4]


def test_cold_batch_merges_mixed_text_and_mrope_position_ids():
    inputs_embeds, prompt_kwargs = ar_module._merge_prefill_prompt_kwargs(
        [
            {
                "inputs_embeds": mx.ones((1, 2, 3)),
                "position_ids": mx.array([[4, 5]], dtype=mx.int32),
            },
            {
                "inputs_embeds": mx.ones((1, 3, 3)) * 2,
                "position_ids": mx.ones((3, 1, 3), dtype=mx.int32) * 7,
            },
        ],
        [[1, 2], [3, 4, 5]],
    )

    assert inputs_embeds.shape == (2, 3, 3)
    assert prompt_kwargs["position_ids"].shape == (3, 2, 3)
    assert prompt_kwargs["position_ids"][0, 0].tolist() == [0, 4, 5]
    assert prompt_kwargs["position_ids"][1, 0].tolist() == [0, 4, 5]
    assert prompt_kwargs["position_ids"][2, 0].tolist() == [0, 4, 5]
    assert prompt_kwargs["position_ids"][0, 1].tolist() == [7, 7, 7]


def test_prompt_processing_batch_slices_native_mrope_position_ids():
    batch = object.__new__(PromptProcessingBatch)
    position_ids = mx.arange(3 * 2 * 5, dtype=mx.int32).reshape(3, 2, 5)
    batch._prompt_kwargs = {"position_ids": position_ids}
    batch._prompt_length_aware_keys = ["position_ids"]

    step_kwargs = batch._prompt_kwargs_for_step(2)

    assert step_kwargs["position_ids"].shape == (3, 2, 2)
    assert step_kwargs["position_ids"].tolist() == position_ids[:, :, :2].tolist()


@pytest.mark.parametrize("budget,chunk_size", [(None, None), (8, 4)])
def test_mixed_apc_batch_strips_private_kwargs_before_prefill(budget, chunk_size):
    bg = object.__new__(BatchGenerator)
    bg.apc_manager = object()
    bg.model = SimpleNamespace(layers=[object()])
    bg.prefill_step_size = budget
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
                "_apc_semantic_hash": 7,
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
            ar_module._apc, "make_warm_batch_kv_cache_multi", return_value=([], 4)
        ),
        patch.object(generate_module, "PromptProcessingBatch", fake_prompt_batch),
    ):
        batch = bg._build_mixed_prompt_batch(sequences)

    assert batch is not None
    assert captured["prefill_step_size"] == chunk_size
    assert "_apc_tenant" not in captured["prompt_kwargs"]
    assert "_apc_image_hash" not in captured["prompt_kwargs"]
    assert "_apc_semantic_hash" not in captured["prompt_kwargs"]
    assert captured["prompt_kwargs"]["keep_tensor"].shape == (2, 1)


def test_apc_pick_rejects_image_tokens_and_releases_blocks():
    block_size = 4
    image_token_id = 99
    token_ids = [image_token_id, 1, 2, 3, 4]
    manager = apc_module.APCManager(num_blocks=4, block_size=block_size)
    layer_keys = [mx.ones((1, 1, block_size, 2))]
    layer_values = [mx.ones((1, 1, block_size, 2)) * 2]
    stored = manager.store_kv_blocks(token_ids[:block_size], layer_keys, layer_values)
    manager.release(stored)

    bg = object.__new__(BatchGenerator)
    bg.apc_manager = manager
    bg.model = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id))
    bg._wire_stack = None

    pick = bg._apc_pick_for((1, token_ids, 1, {}, [], None))

    assert pick is None
    assert all(block.ref_cnt == 0 for block in stored)


class TestBatchTurboQuantizedKVStart:
    def _cache_kinds(self, **kwargs):
        from mlx_vlm.generate.ar import _make_cache

        caches = _make_cache(
            MockModel(), [0], kv_bits=3.5, kv_quant_scheme="turboquant", **kwargs
        )
        return [type(c).__name__ for c in caches]

    def test_defers_to_float_below_threshold(self):
        kinds = self._cache_kinds(quantized_kv_start=5000, prefill_length=16)
        assert "BatchTurboQuantKVCache" not in kinds
        assert set(kinds) == {"BatchKVCache"}


class TestTokenizerPaddedBatchRows:
    """BatchGenerator canonicalizes masked dense rows before queueing them."""

    def _generator(self):
        gen = object.__new__(BatchGenerator)
        gen.max_tokens = 4
        gen.logits_processors = []
        gen._unprocessed_sequences = []
        gen.uid_count = 0
        gen._wire_stack = None
        return gen

    def _batch(self, rows):
        import mlx.nn as nn

        from mlx_vlm.generate.ar import PromptProcessingBatch

        class Tiny(nn.Module):
            def make_cache(self):
                from mlx_vlm.models.cache import ArraysCache, KVCache

                return [KVCache(), ArraysCache(1)]

        return PromptProcessingBatch(
            model=Tiny(),
            uids=list(range(len(rows))),
            input_ids=rows,
            max_tokens=[4] * len(rows),
            inputs_embeds=None,
            prompt_kwargs={},
        )

    def test_unpads_tokens_and_sequence_aligned_prompt_tensors(self):
        input_ids = mx.array([[0, 0, 4, 5], [6, 7, 8, 9], [10, 11, 0, 0]])
        attention_mask = mx.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]])
        inputs_embeds = mx.arange(3 * 4 * 3).reshape(3, 4, 3)
        position_ids = mx.arange(3 * 3 * 4).reshape(3, 3, 4)
        prompt_kwargs = ar_module._split_prompt_kwargs_per_row(
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "rope_deltas": mx.array([[2], [3], [4]]),
            },
            batch_size=3,
        )

        gen = self._generator()
        assert gen.insert(
            input_ids, prompt_kwargs=prompt_kwargs, attention_mask=attention_mask
        ) == [0, 1, 2]
        queued = {sequence[0]: sequence for sequence in gen._unprocessed_sequences}

        assert queued[0][1] == [4, 5]
        assert queued[1][1] == [6, 7, 8, 9]
        assert queued[2][1] == [10, 11]
        assert queued[0][3]["inputs_embeds"].shape == (1, 2, 3)
        assert queued[1][3]["inputs_embeds"].shape == (1, 4, 3)
        assert queued[2][3]["inputs_embeds"].shape == (1, 2, 3)
        assert queued[0][3]["position_ids"].shape == (3, 1, 2)
        assert queued[1][3]["position_ids"].shape == (3, 1, 4)
        assert queued[2][3]["position_ids"].shape == (3, 1, 2)
        assert queued[0][3]["rope_deltas"].tolist() == [[2]]
        assert queued[2][3]["rope_deltas"].tolist() == [[4]]

    def test_rejects_noncontiguous_padding_mask(self):
        gen = self._generator()
        with pytest.raises(ValueError, match="one contiguous prompt span"):
            gen.insert(
                mx.array([[0, 4, 0, 5]]),
                prompt_kwargs=[{"inputs_embeds": mx.ones((1, 4, 3))}],
                attention_mask=mx.array([[0, 1, 0, 1]]),
            )
        assert gen._unprocessed_sequences == []

    def test_accepts_mask_with_existing_python_prompt_rows(self):
        gen = self._generator()

        gen.insert(
            [[0, 4, 5], [6, 7, 8]], attention_mask=mx.array([[0, 1, 1], [1, 1, 1]])
        )

        queued = {sequence[0]: sequence[1] for sequence in gen._unprocessed_sequences}
        assert queued == {0: [4, 5], 1: [6, 7, 8]}


@pytest.fixture
def thinking_processor():
    return ThinkingAwareLogitsProcessor(
        RecordingProcessor(), FakeTokenizer(), enable_thinking=True
    )


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def clone(self):
        clone = RecordingProcessor()
        clone.calls.append(("cloned", None))
        return clone

    def process_last_token(self, token, logits):
        self.calls.append(("process_last_token", token))
        return logits + 1

    def __call__(self, input_ids, logits):
        self.calls.append(("call", input_ids.tolist()))
        return logits + 2


def test_thinking_aware_processor_passes_logits_until_thinking_ends(thinking_processor):
    logits = mx.zeros((1, 3), dtype=mx.float32)
    for token, increment, calls in [
        (11, 0, []),
        (100, 1, [("process_last_token", 100)]),
        (3, 1, [("process_last_token", 100), ("process_last_token", 3)]),
    ]:
        out = thinking_processor.process_last_token(token, logits)
        mx.eval(out)
        assert out.tolist() == (logits + increment).tolist()
        assert thinking_processor.processor.calls == calls


def test_thinking_aware_processor_delegates_immediately_without_thinking():
    processor = ThinkingAwareLogitsProcessor(
        RecordingProcessor(), FakeTokenizer(), enable_thinking=False
    )
    logits = mx.zeros((1, 3), dtype=mx.float32)
    out = processor(mx.array([1, 2, 3]), logits)
    mx.eval(out)
    assert out.tolist() == (logits + 2).tolist()
    assert processor.processor.calls == [("call", [1, 2, 3])]


def test_thinking_aware_processor_clone_resets_phase_state(thinking_processor):
    logits = mx.zeros((1, 3), dtype=mx.float32)
    thinking_processor.process_last_token(100, logits)
    clone = thinking_processor.clone()
    out = clone.process_last_token(11, logits)
    mx.eval(out)
    assert isinstance(clone.processor, RecordingProcessor)
    assert clone.processor is not thinking_processor.processor
    assert clone.processor.calls == [("cloned", None)]
    assert out.tolist() == [[0.0, 0.0, 0.0]]


def test_json_schema_processor_uses_compact_whitespace_pattern(monkeypatch):
    observed = {}
    fake_llguidance = types.ModuleType("llguidance")
    fake_llguidance.__path__ = []
    fake_hf = types.ModuleType("llguidance.hf")

    class FakeJsonCompiler:
        def __init__(self, **kwargs):
            observed["compiler_kwargs"] = kwargs

        def compile(self, schema_text):
            observed["schema_text"] = schema_text
            return "compiled grammar"

    fake_llguidance.JsonCompiler = FakeJsonCompiler
    fake_llguidance.grammar_from = lambda *_args: "fallback grammar"
    fake_hf.from_tokenizer = lambda tokenizer: "llg-tokenizer"
    fake_llguidance.hf = fake_hf
    monkeypatch.setitem(sys.modules, "llguidance", fake_llguidance)
    monkeypatch.setitem(sys.modules, "llguidance.hf", fake_hf)
    structured._llg_tokenizer_cache.clear()

    processor = structured.build_json_schema_logits_processor(
        object(), {"type": "object"}
    )

    assert processor.grammar == "compiled grammar"
    assert observed["compiler_kwargs"] == {
        "separators": (", ", ": "),
        "whitespace_pattern": "",
    }
    assert observed["schema_text"] == '{"type": "object"}'


def _np_p_less_keep(logits, temp):
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    threshold = float((p * p).sum())
    return p >= threshold, np.abs(p - threshold)


def _np_typical_keep(logits, typical_p):
    logp = np.asarray(logits, dtype=np.float64)
    logp = logp - logp.max()
    logp = logp - np.log(np.exp(logp).sum())
    p = np.exp(logp)
    ent = -(p * logp).sum()
    order = np.argsort(np.abs(-logp - ent), kind="stable")
    cum_before_sorted = np.cumsum(p[order]) - p[order]
    cum_before = np.zeros(len(p))
    cum_before[order] = cum_before_sorted
    return cum_before < typical_p, np.abs(cum_before - typical_p)


@pytest.mark.parametrize(
    "name,values,reference,tolerance",
    [
        ("p_less", [0.5, 0.7, 1.0, 1.3, 2.0], _np_p_less_keep, 1e-5),
        ("typical_p", [0.2, 0.5, 0.9, 0.95], _np_typical_keep, 1e-4),
    ],
    ids=["p_less", "typical_p"],
)
def test_sampling_filter_matches_numpy(name, values, reference, tolerance):
    """Compare independent masks away from numerically ambiguous boundaries."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        vocab_size = int(rng.integers(8, 200))
        parameter = float(rng.choice(values))
        logits = rng.normal(0, 3, size=vocab_size).astype(np.float32)
        expected, distance = reference(logits, parameter)
        inputs = mx.array(logits)
        if name == "typical_p":
            inputs = inputs - mx.logsumexp(inputs, axis=-1, keepdims=True)
        output = getattr(sampling, f"apply_{name}")(inputs, parameter)
        mx.eval(output)
        actual = np.isfinite(np.asarray(output.tolist(), dtype=np.float64))
        far = distance > tolerance
        assert np.array_equal(expected[far], actual[far])
    if name == "p_less":
        output = sampling.apply_p_less(mx.array([10.0, 0.0, 0.0, 0.0, 0.0]), 1.0)
        assert (output != -mx.inf).tolist() == [True, False, False, False, False]


@pytest.mark.parametrize(
    "shape,dtype,top_p",
    [
        ((100,), mx.float32, 1.0),
        ((50,), mx.bfloat16, 0.9),
        ((3, 50), mx.bfloat16, 0.9),
    ],
)
def test_top_p_sampling_shape(shape, dtype, top_p):
    tokens = sampling.top_p_sampling(
        mx.zeros(shape, dtype=dtype), top_p=top_p, temperature=1.0
    )
    assert tokens.shape == shape[:-1]


@pytest.mark.parametrize(
    "options,logits,draws,survivors",
    [
        ({"top_n_sigma": 1.0}, [0.0, 1.0, 2.0, 3.0, 4.0], 64, {3, 4}),
        ({"p_less": True}, [10.0, 0.0, 0.0, 0.0, 0.0], 128, {0}),
        ({"typical_p": 0.3}, [10.0, 0.0, 0.0, 0.0, 0.0], 64, {0}),
    ],
    ids=["top_n_sigma", "p_less", "typical_p"],
)
def test_sampler_selects_only_survivors(options, logits, draws, survivors):
    tokens = sampling.make_sampler(temp=1.0, **options)(mx.array([logits] * draws))
    mx.eval(tokens)
    assert set(tokens.tolist()) <= survivors


@pytest.mark.parametrize(
    "name,bad,valid",
    [
        ("top_n_sigma", -1.0, 1.0),
        ("typical_p", 0.0, 0.3),
        ("typical_p", 1.5, 0.3),
        ("min_p", -1.0, 0.1),
        ("top_k", -5, 3),
    ],
)
def test_sampling_validation_does_not_corrupt_compile(name, bad, valid):
    """A parameter error must leave MLX tracing usable by subsequent sampling."""
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    with pytest.raises(ValueError):
        mx.eval(getattr(sampling, f"apply_{name}")(logits, bad))
    tokens = sampling.make_sampler(temp=1.0, **{name: valid})(logits)
    mx.eval(tokens)
    assert tokens.shape == (1,)


@pytest.mark.parametrize(
    "module_name", ["mlx_vlm.generate.ar", "mlx_vlm.server.generation"]
)
@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_positioned_target_sampler_honors_top_k(module_name, top_p):
    sampler = importlib.import_module(module_name)._PositionedTargetSampler(
        temperature=1.0, top_p=top_p, top_k=2, seed=42
    )
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tokens = sampler.sample_target(
        mx.repeat(logprobs, 32, axis=0), row_ids=[0] * 32, positions=list(range(32))
    )
    mx.eval(tokens)
    assert set(tokens.tolist()) <= {2, 3}


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("top_p", [1e-8, 1e-3, 0.7])
def test_top_p_preserves_tied_maxima_and_dtype(dtype, top_p):
    # Include pre-filtered and unnormalized rows: use each row's own mass.
    logprobs = mx.log(mx.array([[0.1, 0.45, 0.45], [0.0, 0.2, 0.8]])) - 0.2
    logprobs = logprobs.astype(dtype)
    filtered = sampling.apply_top_p(logprobs, top_p)
    assert filtered.dtype == dtype
    assert (filtered > -mx.inf).tolist() == [
        [False, True, True],
        [False, False, True],
    ]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("top_p", [1e-8, 1e-3])
def test_tiny_top_p_across_sampling_paths(dtype, top_p):
    logits = mx.array([[0.0, 1.0, 3.0, 2.0]])
    logprobs = (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(dtype)
    assert sampling.make_sampler(temp=1.0, top_p=top_p)(logprobs).tolist() == [2]
    assert sampling.top_p_sampling(logprobs, top_p, 1.0).tolist() == [2]
    for module_name in ("mlx_vlm.generate.ar", "mlx_vlm.server.generation"):
        sampler = importlib.import_module(module_name)._PositionedTargetSampler(
            temperature=1.0, top_p=top_p, seed=42
        )
        assert sampler(logprobs).tolist() == [2]
        assert sampler.sample_target(logprobs, row_ids=[0], positions=[0]).tolist() == [
            2
        ]


@pytest.mark.parametrize(
    "module_name", ["mlx_vlm.generate.ar", "mlx_vlm.server.generation"]
)
def test_positioned_top_p_samples_both_tied_maxima(module_name):
    sampler = importlib.import_module(module_name)._PositionedTargetSampler(
        temperature=1.0, top_p=1e-8, seed=42
    )
    logprobs = mx.log(mx.array([[0.1, 0.45, 0.45]]))
    draws = 128
    samples = sampler.sample_target(
        mx.repeat(logprobs, draws, axis=0),
        row_ids=[0] * draws,
        positions=list(range(draws)),
    )
    assert set(samples.tolist()) == {1, 2}


@pytest.mark.parametrize("temperature", [0.0, 1e-300, 1e-39, 1e-5, 0.009, 0.01, 0.1])
@pytest.mark.parametrize("options", [{}, {"top_p": 0.9}, {"p_less": True}])
def test_sampler_clamps_positive_temperature(temperature, options):
    logprobs = mx.log(mx.array([[0.1, 0.449, 0.451]] * 128))
    effective = 0.01 if 0 < temperature < 0.01 else temperature
    mx.random.seed(42)
    actual = sampling.make_sampler(temp=temperature, **options)(logprobs)
    mx.eval(actual)
    mx.random.seed(42)
    expected = sampling.make_sampler(temp=effective, **options)(logprobs)
    assert actual.tolist() == expected.tolist()
    if temperature > 0 and not options.get("p_less"):
        assert set(actual.tolist()) == {1, 2}
    if temperature == 0:
        assert set(actual.tolist()) == {2}


@pytest.mark.parametrize("seed", [None, 42])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("top_p", [0.0, 0.9])
@pytest.mark.parametrize("temperature", [1e-39, 1e-5])
def test_generate_step_clamps_temperature(seed, dtype, top_p, temperature):
    model = MagicMock()
    model.language_model.return_value = LanguageModelOutput(
        logits=mx.array([[[-4.0, 0.0, 0.003]]], dtype=dtype)
    )
    model.get_input_embeddings.return_value = InputEmbeddingsFeatures(
        inputs_embeds=mx.zeros((1, 1, 4))
    )

    def sample(temp):
        mx.random.seed(42)
        return [
            token
            for token, _ in generate_module.generate_step(
                mx.array([[1]]),
                model,
                pixel_values=None,
                mask=None,
                max_tokens=32,
                temperature=temp,
                top_p=top_p,
                seed=seed,
            )
        ]

    actual = sample(temperature)
    assert actual == sample(0.01)
    assert set(actual) == {1, 2}


# Loading and utility contracts


def test_stopping_criteria_reset():
    class MockProcessor:
        def __init__(self):
            self.tokenizer = type(
                "DummyTokenizer", (), {"pad_token": None, "eos_token": "[EOS]"}
            )()

        def encode(self, text, add_special_tokens=False):
            if "[EOS]" in text:
                return [32008]
            return [1]

    processor = MockProcessor()
    stopping_criteria = StoppingCriteria([2], processor)
    stopping_criteria.add_eos_token_ids("[EOS]")

    stopping_criteria.reset([5, 7])
    assert stopping_criteria.eos_token_ids == [5, 7]
    assert stopping_criteria(7) is True


def test_load_processor_preserves_additional_eos_tokens_on_reset():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_ids=[2]), additional_eos_token_ids=[3]
    )

    class Detokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    with (
        patch("mlx_vlm.utils.AutoProcessor.from_pretrained", return_value=processor),
        patch("mlx_vlm.utils.load_tokenizer", return_value=Detokenizer),
    ):
        loaded = load_processor("unused-model-path")

    criteria = loaded.tokenizer.stopping_criteria
    assert criteria.eos_token_ids == [2, 3]
    criteria.reset([5])
    assert criteria.eos_token_ids == [5, 2, 3]


@pytest.mark.parametrize(("max_tokens", "cache_evals"), [(49, 0), (50, 1), (100, 2)])
def test_generate_step_evaluates_cache_periodically(max_tokens, cache_evals):
    model = MagicMock()
    model.language_model.return_value = LanguageModelOutput(logits=mx.zeros((1, 1, 4)))
    model.get_input_embeddings.return_value = InputEmbeddingsFeatures(
        inputs_embeds=mx.zeros((1, 1, 4))
    )
    cache_state = mx.array([1])

    with patch.object(ar_module.mx, "eval", wraps=mx.eval) as eval_mock:
        list(
            generate_module.generate_step(
                mx.array([[1]]),
                model,
                pixel_values=None,
                mask=None,
                prompt_cache=[SimpleNamespace(state=cache_state)],
                max_tokens=max_tokens,
                temperature=0,
            )
        )

    assert eval_mock.call_count == 1 + cache_evals
    if cache_evals:
        eval_mock.assert_called_with([cache_state])


class TestPrefillScheduling:
    def test_length_buckets_serve_oldest_request_without_starvation(
        self, mock_processor
    ):
        gen = BatchGenerator(FixedLogitModel(), mock_processor)
        try:
            first, peer, short = gen.insert([[1] * 8192, [1] * 7800, [1] * 401])
            selected = gen._take_prefill_sequences(1)
            assert [s[0] for s in selected] == [first]
            # New short requests cannot keep the older long peer waiting.
            gen.insert([[1] * 400] * 8)
            selected = gen._take_prefill_sequences(8)
            assert [s[0] for s in selected] == [peer]
            selected = gen._take_prefill_sequences(2)
            assert selected[0][0] == short
            assert all(len(s[1]) <= 512 for s in selected)
        finally:
            gen.close()
