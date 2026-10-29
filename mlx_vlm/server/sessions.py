"""In-process inference ownership and replay across disconnected streams.

The GPU worker owns the decoder. Readers own only a cursor and a lease: closing
an old HTTP stream cannot pause its replacement, and output produced ahead of
the client remains available without rolling back recurrent or KV caches.
"""

import time
from threading import Condition, Lock


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
                self.history.append(token)
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
