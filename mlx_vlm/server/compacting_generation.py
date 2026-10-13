"""Pause at a context threshold, compact, and resume one logical generation."""

import asyncio
import uuid
from contextlib import aclosing
from dataclasses import replace
from types import SimpleNamespace

from . import compaction
from .generation import GenerationMetrics, _count_prompt_tokens
from .runtime import runtime


async def _inference_call(function, *args, cleanup=None, **kwargs):
    """Close a late worker result when its HTTP consumer disconnects."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if cleanup is not None:

            def finished(future):
                if not future.cancelled() and future.exception() is None:
                    cleanup(future.result())

            task.add_done_callback(finished)
        raise


class CompactingGeneration:
    def __init__(
        self,
        request,
        items,
        model,
        processor,
        config,
        tenant,
        args,
        *,
        render,
        build_gen_args,
        apply_chat_template,
        generate,
        stream_generate,
        prepare_inputs,
        covered=frozenset(),
        tool_module=None,
        generate_kwargs=None,
    ):
        self.request, self.items = request, items
        self.model, self.processor, self.config = model, processor, config
        self.tenant, self.args, self.render = tenant, args, render
        self.build_gen_args = build_gen_args
        self.apply_chat_template = apply_chat_template
        self.generate, self.stream_generate = generate, stream_generate
        self.prepare_inputs, self.covered = prepare_inputs, covered
        self.tool_module = tool_module
        self.generate_kwargs = generate_kwargs or {}
        self.limit = compaction._context_limit(config)
        self.threshold = compaction.automatic_threshold(request, self.limit)
        self.max_tokens = args.max_tokens
        self.output_tokens = self.input_tokens = 0
        self.full_text = self.continuation = ""
        self.metrics = GenerationMetrics()
        self.finish_reason = None
        self.prepared = False

    async def count(self, prompt, images):
        if runtime.response_generator is not None:
            raw = await asyncio.to_thread(
                runtime.response_generator._cpu_preprocess, prompt, images or None, None
            )
        else:
            raw = await asyncio.to_thread(
                self.prepare_inputs,
                self.processor,
                prompts=prompt,
                images=images or None,
                image_token_index=getattr(self.config, "image_token_index", None),
            )
        return _count_prompt_tokens(raw)

    async def compact(self):
        item_id = None
        tail = compaction.continuation_tail(
            self.continuation,
            self.processor,
            self.args,
            getattr(self.tool_module, "tool_call_start", None),
            getattr(self.tool_module, "tool_call_end", None),
        )
        async with aclosing(
            compaction.stream_response_context(
                self.request,
                self.items,
                self.model,
                self.processor,
                self.config,
                self.tenant,
                build_gen_args=self.build_gen_args,
                apply_chat_template=self.apply_chat_template,
                generate=self.generate,
                automatic=True,
                covered=self.covered,
                continuation=self.continuation,
                retained_continuation=tail,
            )
        ) as updates:
            async for update in updates:
                if isinstance(update, compaction.CompactedContext):
                    self.prepared = True
                    self.items, self.covered = update.items, update.covered
                    self.continuation = update.continuation
                    if update.changed:
                        # The capsule follows already-emitted output items. Carry
                        # their retained assistant prefix so replay does not lose
                        # the response produced before this compaction boundary.
                        replay = self.items + (
                            [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": self.continuation,
                                }
                            ]
                            if self.continuation
                            else []
                        )
                        capsule = compaction.seal(
                            replay,
                            model=self.request.model,
                            tenant=self.tenant,
                            covered=self.covered,
                        )
                        if item_id:
                            capsule["id"] = item_id
                        yield {"kind": "compaction", "item": capsule}
                elif update["type"] == "mlx.compaction.started":
                    item_id = f"cmp_{uuid.uuid4().hex}"
                    yield {
                        "kind": "compaction_start",
                        "item": {
                            "type": "compaction",
                            "id": item_id,
                            "encrypted_content": "",
                        },
                    }

    async def tokens(self):
        if not self.prepared:
            async with aclosing(self.compact()) as updates:
                async for event in updates:
                    yield event
        while self.output_tokens < self.max_tokens:
            prompt, images = self.render(self.items)
            prompt += self.continuation
            prompt_tokens = await self.count(prompt, images)
            remaining = self.max_tokens - self.output_tokens
            # A protected input may already exceed the preferred threshold.
            # It may still generate up to capacity; the next pause must reduce it.
            boundary = self.threshold if prompt_tokens < self.threshold else self.limit
            segment_limit = min(remaining, boundary - prompt_tokens)
            if segment_limit <= 0:
                raise compaction.ContextBudgetError(
                    "Protected input leaves no room for generation."
                )
            args = replace(self.args, max_tokens=segment_limit)
            iterator = None
            segment_tokens = 0
            segment_finish = None
            self.input_tokens += prompt_tokens
            try:
                if runtime.response_generator is not None:
                    _, iterator = await _inference_call(
                        runtime.response_generator.generate,
                        cleanup=lambda result: result[1].close(),
                        prompt=prompt,
                        images=images or None,
                        args=args,
                    )
                elif not self.request.stream:
                    result = await asyncio.to_thread(
                        self.generate,
                        model=self.model,
                        processor=self.processor,
                        prompt=prompt,
                        image=images,
                        vision_cache=runtime.model_cache.get("vision_cache"),
                        apc_manager=runtime.apc_manager,
                        **args.to_generate_kwargs(),
                        **self.generate_kwargs,
                    )
                    iterator = iter([result])
                else:
                    iterator = self.stream_generate(
                        model=self.model,
                        processor=self.processor,
                        prompt=prompt,
                        image=images,
                        vision_cache=runtime.model_cache.get("vision_cache"),
                        apc_manager=runtime.apc_manager,
                        **args.to_generate_kwargs(),
                        **self.generate_kwargs,
                    )

                def next_chunk():
                    return next(iterator, None)

                while (
                    chunk := await _inference_call(
                        next_chunk,
                        cleanup=lambda _: getattr(iterator, "close", lambda: None)(),
                    )
                ) is not None:
                    if not hasattr(chunk, "text"):
                        continue
                    count = getattr(chunk, "token_count", None)
                    if count is None:
                        count = max(
                            0, getattr(chunk, "generation_tokens", 0) - segment_tokens
                        )
                    segment_tokens += count
                    self.output_tokens += count
                    self.full_text += chunk.text
                    self.continuation += chunk.text
                    segment_finish = getattr(chunk, "finish_reason", None)
                    if segment_finish is None and segment_tokens >= segment_limit:
                        segment_finish = "length"
                    final = (
                        segment_finish == "stop"
                        or self.output_tokens >= self.max_tokens
                    )
                    # Segment length is an internal pause, not an end-of-message.
                    token = SimpleNamespace(**vars(chunk))
                    token.token_count = count
                    token.finish_reason = segment_finish if final else None
                    self.metrics.record_chunk(token)
                    yield token
                    if segment_finish or segment_tokens >= segment_limit:
                        break
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except ValueError:
                        # A fallback generator may still be executing in the
                        # cancelled next() call; its completion callback closes it.
                        pass
            if segment_finish != "length" or self.output_tokens >= self.max_tokens:
                self.finish_reason = segment_finish or "stop"
                return
            before = prompt_tokens + segment_tokens
            async with aclosing(self.compact()) as updates:
                async for event in updates:
                    yield event
            next_prompt, next_images = self.render(self.items)
            after = await self.count(next_prompt + self.continuation, next_images)
            if after >= before or after >= self.threshold:
                raise compaction.ContextBudgetError(
                    "Compaction did not free space below the threshold; generation cannot continue."
                )
        self.finish_reason = "length"
