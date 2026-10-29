"""Continue a stopped generation without introducing another conversation turn.

Checkpoints refer to the raw prefix consumed by the response parser, rather than
the GPU's (potentially further ahead) output queue. Replaying that prefix seeds
the reasoning and tool parsers; only text already delivered to the client is
suppressed. Tool arguments are replayed in full for the new stream accumulator.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from threading import Lock

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from .schemas import ContinuationOptions
from .sessions import InferenceSession, SessionOutput


@dataclass
class Generation:
    body: dict
    tenant: str
    path: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt: str | None = None
    initial_prefix: str = ""
    parts: list[str] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    session: InferenceSession | None = None
    lease: int = 0


class Checkpoints:
    # Prompt/media references live only in this server process. Bound retention
    # so abandoned chats cannot keep an unbounded amount of context alive.
    def __init__(self, capacity=32, ttl=600):
        self.capacity, self.ttl = capacity, ttl
        self.records = OrderedDict()
        self.lock = Lock()

    def add(self, record):
        with self.lock:
            self.records[record.id] = record
            while len(self.records) > self.capacity:
                _, old = self.records.popitem(last=False)
                if old.session is not None and not any(
                    r.session is old.session for r in self.records.values()
                ):
                    old.session.dispose()

    def resume(self, checkpoint, tenant, path):
        with self.lock:
            old = self.records.get(checkpoint.get("id"))
            if (
                old is None
                or old.tenant != tenant
                or old.path != path
                or time.monotonic() - old.touched > self.ttl
            ):
                raise ValueError(
                    "This response checkpoint is no longer available. Regenerate the response to continue."
                )
            cursor = checkpoint.get("cursor")
            if not isinstance(cursor, int) or not 0 <= cursor <= len(old.parts):
                raise ValueError("Invalid response checkpoint.")
            size = len(old.initial_prefix) + sum(map(len, old.parts[:cursor]))
            if any(
                not isinstance(checkpoint.get(key), int)
                or not 0 <= checkpoint[key] <= size
                for key in ("content", "reasoning")
            ):
                raise ValueError("Invalid response checkpoint text offsets.")
            if old.prompt is None:
                raise ValueError(
                    "The interrupted response has not started generating yet."
                )
            if old.session is None:
                raise ValueError(
                    "The inference checkpoint is no longer available. Regenerate the response."
                )
            lease = old.session.claim()
            record = copy.copy(old)
            record.lease = lease
            record.id = uuid.uuid4().hex
            record.parts = old.parts[:cursor]
            record.touched = time.monotonic()
            # Keep the previous handle usable if the replacement connection
            # drops before delivering its first checkpoint. A new lease moves
            # the reader; it never forks the decoder or duplicates GPU work.
            self.records.move_to_end(old.id)
        self.add(record)
        return record


CHECKPOINTS = Checkpoints()
ACTIVE = ContextVar("mlx_vlm_continuation", default=None)


class StreamState:
    def __init__(self, record, checkpoint=None):
        self.record = record
        self.closed = False
        self.resuming = checkpoint is not None
        self.prefix = (
            record.initial_prefix + "".join(record.parts) if self.resuming else ""
        )
        self.previous = {
            key: int((checkpoint or {}).get(key, 0)) for key in ("content", "reasoning")
        }
        self.remaining = dict(self.previous)
        self.delivered = dict(self.previous)

    def delta(self, kind, value):
        if not isinstance(value, str):
            return value
        count = min(self.remaining[kind], len(value))
        self.remaining[kind] -= count
        value = value[count:]
        self.delivered[kind] += len(value)
        return value

    def checkpoint(self):
        return {
            "id": self.record.id,
            "cursor": len(self.record.parts),
            "content": self.delivered["content"],
            "reasoning": self.delivered["reasoning"],
            "responses": self.record.path.endswith("/responses"),
        }

    def event(self, event):
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta:
                delta["content"] = self.delta("content", delta["content"])
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            reasoning = self.delta("reasoning", reasoning)
            for key in ("reasoning", "reasoning_content"):
                if key in delta:
                    delta[key] = reasoning
        kind = event.get("type")
        if kind in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        ):
            event["delta"] = self.delta(
                "content" if kind == "response.output_text.delta" else "reasoning",
                event.get("delta"),
            )
        if any(self.previous.values()):
            if kind in (
                "response.output_text.done",
                "response.reasoning_text.done",
                "response.reasoning_summary_text.done",
            ):
                key = "content" if kind == "response.output_text.done" else "reasoning"
                event["text"] = event.get("text", "")[self.previous[key] :]
            elif kind == "response.output_item.done":
                self.trim_output([event["item"]])
            elif kind == "response.content_part.done" and "text" in event.get(
                "part", {}
            ):
                event["part"]["text"] = event["part"]["text"][
                    self.previous["content"] :
                ]
        if kind in ("response.completed", "response.incomplete") and any(
            self.previous.values()
        ):
            # The terminal response is also consumed by clients as fallback text.
            # Leave complete tool calls intact but return only the new prose.
            response = event.get("response", {})
            self.trim_output(response.get("output", []))
            if isinstance(response.get("output_text"), str):
                response["output_text"] = response["output_text"][
                    self.previous["content"] :
                ]
        if self.record.prompt is not None:
            event["continuation"] = self.checkpoint()
        return event

    def trim_output(self, items):
        remaining = dict(self.previous)
        for item in items:
            key = "reasoning" if item.get("type") == "reasoning" else "content"
            for part in item.get("summary" if key == "reasoning" else "content", []):
                if "text" not in part:
                    continue
                text = part["text"]
                count = min(remaining[key], len(text))
                part["text"] = text[count:]
                remaining[key] -= count


class ContinuationMiddleware:
    """ASGI send runs before the next parser step, keeping checkpoints atomic."""

    def __init__(self, app, authenticate, read_tenant):
        self.app = app
        self.authenticate, self.read_tenant = authenticate, read_tenant

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or path
            not in (
                "/chat/completions",
                "/v1/chat/completions",
                "/responses",
                "/v1/responses",
            )
        ):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        raw_body = b"".join(chunks)
        try:
            body = json.loads(raw_body)
        except (ValueError, TypeError):
            body = {}
        state = None
        if isinstance(body, dict) and any(
            body.get(key)
            for key in ("resumable", "continuation", "continue_final_message")
        ):
            connection = HTTPConnection(scope)
            try:
                self.authenticate(connection)
                options = ContinuationOptions.model_validate(body)
                if body.get("stream") is not True:
                    raise ValueError("Continuation requires stream=true.")
                if options.continuation is not None and options.continue_final_message:
                    raise ValueError(
                        "Use either continuation or continue_final_message, not both."
                    )
            except HTTPException as error:
                return await JSONResponse(
                    {"detail": error.detail},
                    status_code=error.status_code,
                    headers=error.headers,
                )(scope, receive, send)
            except (ValueError, ValidationError) as error:
                return await JSONResponse(
                    {"error": {"message": str(error)}}, status_code=400
                )(scope, receive, send)
            tenant = self.read_tenant(connection)
            checkpoint = (
                options.continuation.model_dump() if options.continuation else None
            )
            if checkpoint is not None:
                try:
                    record = CHECKPOINTS.resume(checkpoint, tenant, path)
                except (ValueError, TypeError, AttributeError) as error:
                    return await JSONResponse(
                        {"error": {"message": str(error)}}, status_code=409
                    )(scope, receive, send)
                body = copy.deepcopy(record.body)
                # The saved prompt already includes the exact compacted context.
                body["context_management"] = []
                raw_body = json.dumps(body).encode()
            else:
                if body.get("continue_final_message"):
                    body.pop("transient_context", None)
                    body["context_management"] = []
                    raw_body = json.dumps(body).encode()
                record = Generation(copy.deepcopy(body), tenant, path)
                CHECKPOINTS.add(record)
            state = StreamState(record, checkpoint)
        delivered_body = False

        async def replay_receive():
            nonlocal delivered_body
            if not delivered_body:
                delivered_body = True
                return {"type": "http.request", "body": raw_body}
            return await receive()

        buffer = b""
        streaming = False

        async def checkpoint_send(message):
            nonlocal buffer, streaming
            if message["type"] == "http.response.start":
                streaming = b"text/event-stream" in dict(
                    message.get("headers", [])
                ).get(b"content-type", b"")
                if state is not None and streaming:
                    message = {
                        **message,
                        "headers": [
                            *message.get("headers", []),
                            (b"x-mlx-continuation", b"1"),
                        ],
                    }
            if (
                state is not None
                and streaming
                and message["type"] == "http.response.body"
            ):
                buffer += message.get("body", b"")
                blocks = []
                while b"\n\n" in buffer:
                    block, buffer = buffer.split(b"\n\n", 1)
                    lines = block.split(b"\n")
                    for index, line in enumerate(lines):
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            event = json.loads(line[5:])
                            lines[index] = (
                                b"data: "
                                + json.dumps(
                                    state.event(event), ensure_ascii=False
                                ).encode()
                            )
                    blocks.append(b"\n".join(lines) + b"\n\n")
                if not message.get("more_body"):
                    blocks.append(buffer)
                    buffer = b""
                message = {**message, "body": b"".join(blocks)}
            await send(message)

        token = ACTIVE.set(state)
        try:
            scope = {
                **scope,
                "headers": [
                    (key, value)
                    for key, value in scope.get("headers", [])
                    if key != b"content-length"
                ]
                + [(b"content-length", str(len(raw_body)).encode())],
            }
            await self.app(scope, replay_receive, checkpoint_send)
        finally:
            if state is not None:
                state.closed = True
                if state.record.session is not None:
                    state.record.session.pause(state.record.lease)
            ACTIVE.reset(token)


def render_prompt(template, *args, **kwargs):
    state = ACTIVE.get()
    if state is not None and state.resuming:
        return state.record.prompt
    if state is not None and state.record.body.get("continue_final_message"):
        # Older transcripts have parsed text but no raw checkpoint. Cut the
        # final assistant turn at a marker inside the field being continued;
        # do not render its closing reasoning/turn delimiters.
        processor, config, messages, *rest = args
        messages = copy.deepcopy(messages)
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Continuation requires an unfinished assistant message.")
        last = messages[-1]
        content = last.get("content") or ""
        reasoning = next(
            (
                last[key]
                for key in ("reasoning", "reasoning_content", "thinking")
                if last.get(key)
            ),
            "",
        )
        base = template(
            processor,
            config,
            messages[:-1],
            *rest,
            **{**kwargs, "add_generation_prompt": True},
        )
        marker = "MLX_CONTINUATION_" + uuid.uuid4().hex
        key = "content"
        if not last.get("content"):
            key = next(
                (
                    key
                    for key in ("reasoning", "reasoning_content", "thinking")
                    if last.get(key)
                ),
                "content",
            )
        last[key] = (last.get(key) or "") + marker
        kwargs.update(add_generation_prompt=False, preserve_thinking=True)
        rendered = template(processor, config, messages, *rest, **kwargs)
        if marker not in rendered:
            raise ValueError(
                "This model's template cannot continue the interrupted message."
            )
        rendered = rendered[: rendered.rindex(marker)]
        # Replay the old text through the parser as well. Otherwise a new
        # response parser trims the first space and joins "open" + "the".
        boundary = 0
        for left, right in zip(base, rendered):
            if left != right:
                break
            boundary += 1
        state.prefix = rendered[boundary:]
        state.record.initial_prefix = state.prefix
        state.previous = {"content": len(content), "reasoning": len(reasoning)}
        state.remaining = dict(state.previous)
        state.delivered = dict(state.previous)
        return rendered[:boundary]
    return template(*args, **kwargs)


def prepare_generation(prompt, images, audio, args, videos):
    state = ACTIVE.get()
    if state is None:
        return prompt, images, audio, args, videos
    record = state.record
    if state.resuming:
        raise RuntimeError(
            "Resume must restore the retained decoder, not rebuild its prompt."
        )
    record.prompt = prompt
    prompt += state.prefix
    args = copy.copy(args)
    args.skip_special_tokens = False
    return prompt, images, audio, args, videos


class ContinuationTokens:
    def __init__(self, iterator, state, prefix_token):
        self.iterator, self.state = iterator, state
        self.prefix_token = prefix_token if state.prefix else None
        cursor = len(state.record.parts)
        if self.prefix_token is not None and cursor:
            previous = state.record.session.history[cursor - 1]
            # Restore cumulative measurements alongside parser state. Replaying
            # the acknowledged prefix performs no additional generation.
            self.prefix_token = SessionOutput(
                replace(
                    previous.value,
                    text=state.prefix,
                    token=prefix_token.token,
                    logprobs=prefix_token.logprobs,
                    finish_reason=None,
                ),
                previous.decode_progress,
                0,
                previous.emitted_at,
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.prefix_token is not None:
            token, self.prefix_token = self.prefix_token, None
            return token
        token = next(self.iterator)
        if token is not None and hasattr(token, "text"):
            self.state.record.parts.append(token.text)
            self.state.record.touched = time.monotonic()
        return token

    def close(self):
        close = getattr(self.iterator, "close", None)
        if close is not None:
            close()


def start_generation(generate, prompt, images=None, audio=None, args=None, videos=None):
    """Restore the same decoder and replay only to seed the protocol parsers."""
    state = ACTIVE.get()
    if state is None:
        return generate(
            prompt, images, audio, args=args, **({"videos": videos} if videos else {})
        )
    from .generation import StreamingToken, get_token_queue_timeout

    if state.resuming:
        session = state.record.session
        context = session.context
        iterator = session.reader(
            len(state.record.parts), state.record.lease, get_token_queue_timeout()
        )
    else:
        prepared = prepare_generation(prompt, images, audio, args, videos)
        session = InferenceSession(ttl=CHECKPOINTS.ttl)
        session.owner = getattr(generate, "__self__", None)
        state.record.session = session
        try:
            if state.closed:
                raise RuntimeError("The stream closed before generation started.")
            context, iterator = generate(*prepared, session=session)
        except Exception as error:
            session.dispose(str(error))
            raise
        session.context = context
    prefix = StreamingToken(
        text=state.prefix, token=0, logprobs=0, finish_reason=None, token_count=0
    )
    return context, ContinuationTokens(iterator, state, prefix)


def stream_generation(
    generate, *, prompt, image=None, audio=None, video=None, args, **kwargs
):
    """The non-batched backend follows the same checkpoint and parser contract."""
    state = ACTIVE.get()
    if state is None:
        return generate(
            prompt=prompt,
            image=image,
            audio=audio,
            video=video,
            **args.to_generate_kwargs(),
            **kwargs,
        )
    from ..generate import GenerationResult

    if state.resuming:
        session = state.record.session
    else:
        from .generation import ResponseGenerator

        prompt, image, audio, args, video = prepare_generation(
            prompt, image, audio, args, video
        )
        # The retained generator frame owns the cache and decode position.
        # Keyed sampling also keeps other requests from perturbing its draws.
        sampler = ResponseGenerator._make_session_sampler(args)
        iterator = generate(
            prompt=prompt,
            image=image,
            audio=audio,
            video=video,
            skip_special_tokens=False,
            sampler=sampler,
            **args.to_generate_kwargs(),
            **kwargs,
        )
        session = InferenceSession(ttl=CHECKPOINTS.ttl, source=iterator)
        state.record.session = session
    iterator = session.reader(len(state.record.parts), state.record.lease)
    return ContinuationTokens(iterator, state, GenerationResult(text=state.prefix))
