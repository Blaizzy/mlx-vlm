"""Bounded continuous prefill scheduling shared by the decision CLI and server."""

import queue
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Event, Thread, current_thread
from types import SimpleNamespace

from ._stream_cleanup import clear_mlx_streams
from .decision import DecisionCancelled, check_cancelled, normalize_questions


class PredictionEngine:
    """Fallback for models that expose predict but no incremental engine.

    Cancellation is checked around each model call; a running call cannot be
    interrupted unless the model supplies an incremental decision engine.
    """

    def __init__(self, model, processor, *, max_length=None, **kwargs):
        self.model, self.processor, self.max_length = model, processor, max_length

    def prepare(self, request, *, namespace=None, cancelled=None):
        cancelled = cancelled if cancelled is not None else Event()
        check_cancelled(cancelled)
        return SimpleNamespace(
            request=request,
            cancelled=cancelled,
            offset=0,
            done=False,
            cached_tokens=0,
            result=None,
            error=None,
        )

    def step(self, states):
        for state in states:
            if state.cancelled.is_set():
                continue
            body = state.request
            kwargs = {
                k: v
                for k, v in body.items()
                if k not in ("model", "state", "questions")
            }
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            try:
                state.result = self.model.predict(
                    self.processor, body.get("state"), body["questions"], **kwargs
                )
            except Exception as error:
                state.error = error
            state.done = True
            state.offset = 1

    def finish(self, state):
        check_cancelled(state.cancelled)
        if not state.done:
            raise ValueError("Cannot finish an incomplete decision")
        if state.error is not None:
            raise state.error
        return {"response": state.result, "probabilities": None, "cached_tokens": 0}


def make_decision_engine(model, processor, **settings):
    """Select a model-owned incremental backend or ordinary prediction.

    Backends implement prepare(request, namespace, cancelled), step(states), and
    finish(state). States expose offset/done for admission between bounded steps;
    finish returns a native response, optional unrounded probabilities by question
    and label, and cached_tokens. This scheduler never accesses model internals.
    """
    factory = getattr(model, "make_decision_engine", None)
    if factory is not None:
        return factory(processor, **settings)
    return PredictionEngine(model, processor, **settings)


class DecisionQueueFull(Exception):
    pass


@dataclass
class DecisionJob:
    request: dict
    namespace: object = None
    allow_single_criterion: bool = False
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
        self.worker = Thread(target=self._run, name="decision-inference", daemon=True)
        self.worker.start()

    def submit(self, request, *, namespace=None, allow_single_criterion=False):
        if self.stopping.is_set():
            raise DecisionCancelled("Decision scheduler stopped")
        if not self.slots.acquire(blocking=False):
            raise DecisionQueueFull("Decision request queue is full")
        job = DecisionJob(request, namespace, allow_single_criterion)
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
                            self.engine = make_decision_engine(
                                model,
                                processor,
                                max_length=self.max_length,
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
                        body = dict(job.request)
                        questions, media = normalize_questions(
                            self.engine.model,
                            body["questions"],
                            allow_single_criterion=job.allow_single_criterion,
                            **{
                                k: v
                                for k, v in body.items()
                                if k not in ("model", "state", "questions")
                            },
                        )
                        body = {
                            "model": body["model"],
                            "state": body.get("state"),
                            "questions": questions,
                            **media,
                        }
                        state = self.engine.prepare(
                            body, namespace=job.namespace, cancelled=job.cancelled
                        )
                        active.append((job, body, state))
                    except Exception as exc:
                        _complete(job, error=exc)
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
                        output = self.engine.finish(state)
                        output["response"]["model"] = body["model"]
                        _complete(job, result=output)
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
            clear_mlx_streams()
            self.active_count = self.pending_count = 0
