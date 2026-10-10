"""Bounded continuous prefill scheduling shared by the decision CLI and server."""

import queue
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Event, Thread, current_thread

from .decision import decision_probabilities, format_response, prepare_request
from .models.clef.inference import (
    DecisionCancelled,
    DecisionEngine,
    check_cancelled,
    context_limit,
)


class DecisionQueueFull(Exception):
    pass


@dataclass
class DecisionJob:
    request: dict
    namespace: object = None
    future: Future = field(default_factory=Future)
    cancelled: Event = field(default_factory=Event)

    def cancel(self):
        self.cancelled.set()
        self.future.cancel()


def _complete(job, *, result=None, error=None):
    try:
        if error is None:
            job.future.set_result(result)
        else:
            job.future.set_exception(error)
    except InvalidStateError:
        pass  # A waiting client cancelled its future concurrently.


class DecisionScheduler:
    def __init__(
        self,
        loader,
        *,
        max_length=None,
        batch_size=4,
        prefill_step_size=512,
        cache_bytes=256 * 1024**2,
        max_pending=64,
        batch_wait_ms=5,
    ):
        if (
            batch_size < 1
            or prefill_step_size < 1
            or max_pending < 1
            or cache_bytes < 0
            or batch_wait_ms < 0
        ):
            raise ValueError("Invalid decision scheduler limits")
        self.loader = loader
        self.max_length = max_length
        self.batch_size = batch_size
        self.prefill_step_size = prefill_step_size
        self.cache_bytes = cache_bytes
        self.batch_wait_ms = batch_wait_ms
        self.queue = queue.Queue(maxsize=max_pending)
        self.slots = BoundedSemaphore(max_pending + batch_size)
        self.stopping = Event()
        self.engine = None
        self.active_count = self.pending_count = 0
        self.worker = Thread(target=self._run, name="clef-inference", daemon=True)
        self.worker.start()

    def submit(self, request, *, namespace=None):
        if self.stopping.is_set():
            raise DecisionCancelled("Decision scheduler stopped")
        if not self.slots.acquire(blocking=False):
            raise DecisionQueueFull("Decision request queue is full")
        job = DecisionJob(request, namespace)
        job.future.add_done_callback(lambda future: self.slots.release())
        try:
            self.queue.put_nowait(job)
        except queue.Full as exc:
            job.cancel()
            raise DecisionQueueFull("Decision request queue is full") from exc
        # Cover shutdown racing with put_nowait after the final queue drain.
        if self.stopping.is_set():
            job.cancel()
        return job

    def qsize(self):
        return self.queue.qsize() + self.pending_count

    def is_worker_thread(self):
        return current_thread() is self.worker

    def stop_and_join(self):
        self.stopping.set()
        # Active jobs share the stop signal through their checkpoint callback.
        for job in getattr(self, "_active_jobs", ()):
            job.cancel()
        if not self.is_worker_thread():
            self.worker.join()

    def _run(self):
        pending, active = deque(), []
        current_model = None
        processor = config = None
        try:
            while not self.stopping.is_set():
                if not active and not pending:
                    try:
                        pending.append(self.queue.get(timeout=0.05))
                    except queue.Empty:
                        continue
                    # There is nothing to coalesce when only one row is allowed.
                    if self.batch_size > 1:
                        self.stopping.wait(self.batch_wait_ms / 1000)
                while len(pending) < self.queue.maxsize:
                    try:
                        pending.append(self.queue.get_nowait())
                    except queue.Empty:
                        break
                for job in list(pending):
                    if job.cancelled.is_set() or job.future.cancelled():
                        pending.remove(job)
                        _complete(job, error=DecisionCancelled("Decision cancelled"))
                for item in list(active):
                    if item[0].cancelled.is_set():
                        active.remove(item)
                        _complete(
                            item[0], error=DecisionCancelled("Decision cancelled")
                        )
                self.pending_count, self.active_count = len(pending), len(active)
                if not active and pending:
                    name = pending[0].request["model"]
                    if name != current_model:
                        self.engine = (
                            None  # Release old model and prefix cache before loading.
                        )
                        processor = config = None
                        current_model = None
                        try:
                            model, processor, config = self.loader(name)
                            self.engine = DecisionEngine(
                                model,
                                prefill_step_size=self.prefill_step_size,
                                cache_bytes=self.cache_bytes,
                            )
                            del model
                            current_model = name
                        except Exception as exc:
                            _complete(pending.popleft(), error=exc)
                            continue
                # FIFO model boundaries prevent a stream of one model starving
                # another. New arrivals can occupy slots after every prefill step.
                while (
                    pending
                    and len(active) < self.batch_size
                    and pending[0].request["model"] == current_model
                ):
                    job = pending.popleft()
                    self._active_jobs = tuple(item[0] for item in active) + (job,)
                    if not job.future.set_running_or_notify_cancel():
                        continue
                    try:
                        check_cancelled(self.stopping)
                        limit = context_limit(config, self.max_length)
                        body, encoded = prepare_request(
                            processor,
                            job.request,
                            limit,
                            lambda: check_cancelled(job.cancelled),
                        )
                        state = self.engine.prepare(
                            encoded, namespace=job.namespace, cancelled=job.cancelled
                        )
                        active.append((job, body, state))
                    except Exception as exc:
                        _complete(job, error=exc)
                    finally:
                        encoded = None
                self._active_jobs = tuple(item[0] for item in active)
                self.active_count, self.pending_count = len(active), len(pending)
                if not active:
                    continue
                group = [
                    item for item in active if item[2].offset == active[0][2].offset
                ]
                try:
                    self.engine.step([item[2] for item in group])
                except Exception as exc:
                    for item in group:
                        active.remove(item)
                        _complete(item[0], error=exc)
                    group.clear()
                    item = state = None
                    continue
                for job, body, state in group:
                    active.remove((job, body, state))
                    if not state.done:
                        active.append((job, body, state))
                        continue
                    try:
                        probabilities = decision_probabilities(
                            self.engine.finish(state)
                        )
                        _complete(
                            job,
                            result={
                                "response": format_response(
                                    body, state.record, probabilities
                                ),
                                "probabilities": probabilities,
                                "cached_tokens": state.cached_tokens,
                            },
                        )
                    except Exception as exc:
                        _complete(job, error=exc)
                # Do not retain the last completed request's full hidden/KV
                # buffers in idle worker locals outside the bounded prefix cache.
                group.clear()
                item = state = None
                self._active_jobs = tuple(entry[0] for entry in active)
                self.active_count = len(active)
        finally:
            for job in list(pending) + [item[0] for item in active]:
                _complete(job, error=DecisionCancelled("Decision scheduler stopped"))
            while True:
                try:
                    _complete(
                        self.queue.get_nowait(),
                        error=DecisionCancelled("Decision scheduler stopped"),
                    )
                except queue.Empty:
                    break
            self._active_jobs = ()
            self.engine = None
            self.active_count = self.pending_count = 0
