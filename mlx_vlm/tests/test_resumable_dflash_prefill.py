from queue import Queue
from threading import Lock
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.server import generation


def test_continuation_segments_leave_final_token():
    continuation = generation.DFlashPrefillContinuation(
        request_id="request",
        prompt_cache=[],
        remaining_input_ids=mx.array([[1, 2, 3, 4]]),
        remaining_embeds=mx.zeros((1, 4, 2)),
        prompt_kwargs={"mask": mx.ones((1, 4))},
        cached_tokens=10,
        full_input_ids=list(range(14)),
    )

    assert continuation.next_segment_tokens(10) == 3
    continuation.commit_segment(
        3,
        next_ids=continuation.remaining_input_ids[:, 3:],
        next_embeds=continuation.remaining_embeds[:, 3:],
        next_kwargs={"mask": continuation.prompt_kwargs["mask"][:, 3:]},
    )

    assert continuation.ready_for_final
    assert continuation.processed_uncached == 3
    assert continuation.segment_count == 1


def test_prepare_exception_notifies_submitting_queue(monkeypatch):
    monkeypatch.setattr(generation, "speculative_prefill_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(generation, "_get_draft_block_size_from_env", lambda: 1)
    rqueue = Queue()
    request = SimpleNamespace(rqueue=rqueue, images=[], audio=[], videos=[])

    class Fake:
        _stop = False
        _cancel_lock = Lock()
        _cancelled = set()
        model = SimpleNamespace(language_model=object())
        draft_model = object()

        def __init__(self):
            self.calls = 0

        def _collect_pending_requests(self, **_kwargs):
            self.calls += 1
            return ([request], False) if self.calls == 1 else ([], True)

        def _prepare_resumable_dflash_request(self, *_args):
            raise RuntimeError("synthetic prepare failure")

    generation.ResponseGenerator._run_speculative_resumable_dflash(Fake())

    error = rqueue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "synthetic prepare failure"
    assert rqueue.get_nowait() is None


def test_hot_loop_yields_at_an_evaluated_slice_boundary():
    continuation = SimpleNamespace(ready_for_final=False)
    scheduled = SimpleNamespace(metadata={"continuation": continuation})

    class Fake:
        requests = Queue()
        _cancel_lock = Lock()
        _cancelled = set()

        def __init__(self):
            self.advances = 0

        def _advance_one_resumable_dflash_segment(self, *_args):
            self.advances += 1
            return True

    fake = Fake()
    assert generation.ResponseGenerator._advance_resumable_dflash_segment(
        fake,
        scheduled,
        64,
        object(),
        {},
        object(),
        should_yield=lambda: True,
    )
    assert fake.advances == 1


def test_shutdown_notifies_resident_client(monkeypatch):
    monkeypatch.setattr(generation, "speculative_prefill_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(generation, "_get_draft_block_size_from_env", lambda: 1)
    rqueue = Queue()
    request = SimpleNamespace(rqueue=rqueue, images=[], audio=[], videos=[])
    continuation = SimpleNamespace(
        ready_for_final=False,
        metadata={"request": request},
        processed_uncached=0,
        remaining_tokens=10,
    )
    item = generation.ResumablePrefill(
        request_id="resident",
        work=generation.TailWork(10),
        cache_state=object(),
        remaining_input=continuation,
        metadata={"continuation": continuation},
    )

    class Fake:
        _stop = False
        _cancel_lock = Lock()
        _cancelled = set()
        model = SimpleNamespace(language_model=object())
        draft_model = object()

        def _collect_pending_requests(self, **_kwargs):
            return [request], False

        def _prepare_resumable_dflash_request(self, *_args):
            return item

        def _drain_cancellations(self):
            return set()

        def _advance_resumable_dflash_segment(self, *_args, **_kwargs):
            self._stop = True
            return True

    generation.ResponseGenerator._run_speculative_resumable_dflash(Fake())

    error = rqueue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "generation server stopped"
    assert rqueue.get_nowait() is None


def test_scheduler_cancellation_preserves_foreign_request(monkeypatch):
    monkeypatch.setattr(generation, "speculative_prefill_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(generation, "_get_draft_block_size_from_env", lambda: 1)
    rqueue = Queue()
    request = SimpleNamespace(rqueue=rqueue, images=[], audio=[], videos=[])
    continuation = SimpleNamespace(
        ready_for_final=False,
        metadata={"request": request},
        processed_uncached=0,
        remaining_tokens=10,
    )
    item = generation.ResumablePrefill(
        request_id=str(id(rqueue)),
        work=generation.TailWork(10),
        cache_state=object(),
        remaining_input=continuation,
        metadata={"continuation": continuation},
    )
    foreign_uid = id(rqueue) + 1

    class Fake:
        _stop = False
        _cancel_lock = Lock()
        _cancelled = {id(rqueue), foreign_uid}
        model = SimpleNamespace(language_model=object())
        draft_model = object()

        def __init__(self):
            self.calls = 0

        def _collect_pending_requests(self, **_kwargs):
            self.calls += 1
            return ([request], False) if self.calls == 1 else ([], True)

        def _prepare_resumable_dflash_request(self, *_args):
            return item

    fake = Fake()
    generation.ResponseGenerator._run_speculative_resumable_dflash(fake)

    assert fake._cancelled == {foreign_uid}
    assert rqueue.get_nowait() is None


def test_context_accumulator_keeps_bounded_tail_and_offset():
    first = mx.array([[[1], [2], [3]]])
    second = mx.array([[[4], [5]]])

    combined, dropped = generation._append_dflash_context(first, second, 4, 0)

    assert combined.tolist() == [[[2], [3], [4], [5]]]
    assert dropped == 1


def test_resident_cap_is_bounded_and_honors_global_cap(monkeypatch):
    monkeypatch.delenv("MLX_VLM_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("MLX_VLM_RESUMABLE_PREFILL_MAX_SEQS", raising=False)
    assert generation._resumable_dflash_max_seqs() == 4

    monkeypatch.setenv("MLX_VLM_RESUMABLE_PREFILL_MAX_SEQS", "8")
    monkeypatch.setenv("MLX_VLM_MAX_NUM_SEQS", "3")
    assert generation._resumable_dflash_max_seqs() == 3
