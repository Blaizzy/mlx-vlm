"""Transport keepalives for OpenAI-compatible event streams."""

import asyncio
from contextlib import suppress
from typing import AsyncGenerator

import anyio


async def with_keepalives(
    source: AsyncGenerator[str, None], interval: float
) -> AsyncGenerator[str, None]:
    """Send SSE comments during idle periods without cancelling the producer.

    Keep just one pending read, so a slow client still backpressures generation.
    Closing the response cancels that read and runs the source's cleanup. A zero
    interval disables comments, but retains the same cleanup semantics.
    """
    pending = None
    try:
        while True:
            pending = asyncio.create_task(anext(source))
            while not pending.done():
                ready, _ = await asyncio.wait({pending}, timeout=interval or None)
                if not ready:
                    yield ": keepalive\n\n"
            try:
                ready_task, pending = pending, None
                chunk = ready_task.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        # Starlette cancels the stream inside an AnyIO cancellation scope. Shield
        # cleanup so repeated cancellation cannot interrupt the source's finally.
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending
            await source.aclose()
