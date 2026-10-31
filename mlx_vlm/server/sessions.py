"""In-process inference ownership and replay across disconnected streams.

The GPU worker owns the decoder. Readers own only a cursor and a lease: closing
an old HTTP stream cannot pause its replacement, and output produced ahead of
the client remains available without rolling back recurrent or KV caches.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock


@dataclass(frozen=True)
class DecodeProgress:
    tokens: int = 0
    measured_tokens: int = 0
    elapsed: float = 0.0
    chunk_rate: float | None = None

    @property
    def rate(self):
        return self.measured_tokens / self.elapsed if self.elapsed > 0 else None


@dataclass(frozen=True)
class SessionOutput:
    """A token with immutable producer timings, also used when replaying it."""

    value: object
    decode_progress: DecodeProgress
    token_count: int
    emitted_at: float

    def __getattr__(self, name):
        return getattr(self.value, name)

    @property
    def generation_tokens(self):
        return self.decode_progress.tokens

    @property
    def generation_tps(self):
        return self.decode_progress.rate


class InferenceSession:
    def __init__(self, *, ttl=600, source=None):
        self.condition = Condition()
        self.history = []
        self.context = None
        self.lease = 0
        self.paused = False
        self.finished = False
        self.error = None
        self.touched = time.monotonic()
        self.ttl = ttl
        self.source = source
        self.producer_lock = Lock()
        self.owner = None
        self._generation_seconds = 0.0
        self._step_started_at = None
        self._first_token_at = None
        self._last_token_at = None
        self._first_chunk_tokens = 0
        self.decode_progress = DecodeProgress()

    @property
    def generation_time(self):
        elapsed = self._generation_seconds
        if self._step_started_at is not None:
            elapsed += time.perf_counter() - self._step_started_at
        return elapsed

    @contextmanager
    def generating(self):
        """Count producer work, excluding pauses, replay and other requests.

        A step already in flight when Stop arrives still counts in full.
        The first emitted token establishes the end of prefill.
        """
        self._step_started_at = time.perf_counter()
        try:
            yield
        finally:
            self._generation_seconds = self.generation_time
            self._step_started_at = None

    def _measure(self, token):
        count = getattr(token, "token_count", None)
        if count is None:
            count = max(
                0, getattr(token, "generation_tokens", 0) - self.decode_progress.tokens
            )
        count = max(0, int(count))
        if count:
            now = self.generation_time
            if self._first_token_at is None:
                self._first_token_at = now
                self._first_chunk_tokens = count
            interval = (
                now - self._last_token_at if self._last_token_at is not None else 0
            )
            tokens = self.decode_progress.tokens + count
            self.decode_progress = DecodeProgress(
                tokens=tokens,
                measured_tokens=tokens - self._first_chunk_tokens,
                elapsed=now - self._first_token_at,
                chunk_rate=count / interval if interval > 0 else None,
            )
            self._last_token_at = now
        return SessionOutput(
            token,
            self.decode_progress,
            count,
            getattr(token, "emitted_at", None) or time.perf_counter(),
        )

    def check(self):
        if self.owner is not None and getattr(self.owner, "_stop", False):
            raise ValueError("The model was unloaded. Regenerate the response.")
        if self.error is not None:
            raise ValueError(str(self.error))
        if self.paused and time.monotonic() - self.touched > self.ttl:
            raise ValueError(
                "The inference checkpoint expired. Regenerate the response."
            )

    def claim(self):
        with self.condition:
            self.check()
            self.lease += 1
            self.paused = False
            self.touched = time.monotonic()
            self.condition.notify_all()
            return self.lease

    def pause(self, lease):
        with self.condition:
            if lease == self.lease and not self.paused:
                self.paused = True
                self.touched = time.monotonic()
                self.condition.notify_all()

    def dispose(
        self,
        message="The inference checkpoint is no longer available. Regenerate the response.",
    ):
        with self.condition:
            self.error = ValueError(message)
            self.history.clear()
            self.condition.notify_all()
        # Worker-owned decoders are released by the GPU scheduler. The legacy
        # pull backend has no worker, so close it under its producer lock.
        if self.source is not None:
            with self.producer_lock:
                close = getattr(self.source, "close", None)
                if close is not None:
                    close()
                self.source = None

    def put(self, token):
        with self.condition:
            if self.error is not None:
                return
            if isinstance(token, Exception):
                self.error = token
            elif token is None:
                self.finished = True
            else:
                self.history.append(self._measure(token))
                self.finished = bool(getattr(token, "finish_reason", None))
            self.condition.notify_all()

    def reader(self, cursor=0, lease=None, timeout=300):
        return SessionReader(
            self, cursor, self.lease if lease is None else lease, timeout
        )


class SessionReader:
    def __init__(self, session, cursor, lease, timeout):
        self.session, self.cursor, self.lease = session, cursor, lease
        self.timeout = timeout
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        session = self.session
        if session.source is not None:
            with session.producer_lock:
                if (
                    not self.closed
                    and self.lease == session.lease
                    and self.cursor == len(session.history)
                    and not session.finished
                ):
                    session.check()
                    try:
                        with session.generating():
                            session.put(next(session.source))
                    except StopIteration:
                        session.put(None)
                    if session.finished:
                        close = getattr(session.source, "close", None)
                        if close is not None:
                            close()
                        session.source = None
        with session.condition:
            ready = session.condition.wait_for(
                lambda: self.closed
                or self.lease != session.lease
                or session.error is not None
                or self.cursor < len(session.history)
                or session.finished,
                timeout=self.timeout,
            )
            if self.closed or self.lease != session.lease:
                raise StopIteration
            session.check()
            if not ready:
                self.close()
                raise RuntimeError("Timed out waiting for the next generated token.")
            if self.cursor == len(session.history):
                raise StopIteration
            token = session.history[self.cursor]
            self.cursor += 1
            return token

    def close(self):
        self.closed = True
        self.session.pause(self.lease)
