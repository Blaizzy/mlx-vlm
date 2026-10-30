"""SSE transport lifetime, independent of model/token generation."""

import asyncio

import anyio
import pytest

from mlx_vlm.server.sse import with_keepalives


def test_keepalives_preserve_one_pending_read_and_payload():
    async def check():
        release = asyncio.Event()
        reads, closed = [], []

        async def source():
            try:
                reads.append(1)
                await release.wait()
                yield 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                yield "data: [DONE]\n\n"
            finally:
                closed.append(True)

        stream = with_keepalives(source(), 0.01)
        for _ in range(3):
            assert await anext(stream) == ": keepalive\n\n"
        assert reads == [1]
        release.set()
        assert [chunk async for chunk in stream] == [
            'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        assert closed == [True]

    asyncio.run(check())


@pytest.mark.parametrize("interval", [0, 0.01])
def test_keepalive_disconnect_closes_source(interval):
    async def check():
        started, closed = asyncio.Event(), asyncio.Event()

        async def source():
            try:
                started.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                # Exercise async cleanup under Starlette's cancellation model.
                await asyncio.sleep(0)
                closed.set()

        async def consume():
            async for _ in with_keepalives(source(), interval):
                pass

        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await started.wait()
            group.cancel_scope.cancel()
        assert closed.is_set()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(check())


@pytest.mark.parametrize("interval", [0, 0.01])
def test_keepalive_errors_propagate_after_cleanup(interval):
    async def check():
        closed = []

        async def source():
            try:
                yield "event: response.created\ndata: {}\n\n"
                raise RuntimeError("generation failed")
            finally:
                closed.append(True)

        stream = with_keepalives(source(), interval)
        assert await anext(stream) == "event: response.created\ndata: {}\n\n"
        with pytest.raises(RuntimeError, match="generation failed"):
            await anext(stream)
        assert closed == [True]

    asyncio.run(check())


def test_zero_interval_passes_delayed_output_without_comments():
    async def check():
        async def source():
            await asyncio.sleep(0.03)
            yield "data: [DONE]\n\n"

        assert [chunk async for chunk in with_keepalives(source(), 0)] == [
            "data: [DONE]\n\n"
        ]

    asyncio.run(check())
