"""Bounded streaming inference over image/mask/point-map requests."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .depth import estimate_pointmap
from .flow import LATENTS
from .mesh import extract_mesh, unique
from .processing import decode_pose, prepare_inputs
from .sam3d_objects import Model
from .sparse import Grid, nonzero


@dataclass(frozen=True)
class Request:
    image: mx.array
    mask: Optional[mx.array] = None
    pointmap: Optional[mx.array] = None
    request_id: str = ""
    seed: int = 42
    ss_steps: Optional[int] = None
    slat_steps: Optional[int] = None
    formats: tuple = ("gaussian", "mesh")
    estimate_depth: bool = True


@dataclass(frozen=True)
class Event:
    request_id: str
    stage: str
    step: int
    total_steps: int
    data: dict


def schedule(steps, rescale):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Inference step counts must be positive integers")
    return [
        (i / steps) / (1 + (rescale - 1) * (1 - i / steps)) for i in range(steps + 1)
    ]


def sample(
    flow,
    state,
    steps,
    rescale,
    strength,
    condition,
    uncondition=None,
    *,
    grid=None,
    cancel=None,
):
    times = schedule(steps, rescale)
    if strength > 0 and uncondition is None:
        uncondition = mx.zeros_like(condition)
    for i, (t0, t1) in enumerate(zip(times, times[1:]), 1):
        if cancel is not None and cancel.is_set():
            return
        time = mx.array([t0 * 1000], mx.float32)
        guided = strength > 0 and t0 <= 0.5
        if isinstance(state, dict):
            if guided and hasattr(flow, "guided"):
                velocity, uncond = flow.guided(
                    state, time, condition, uncondition, mx.zeros(1)
                )
            else:
                velocity = flow(state, time, condition, mx.zeros(1))
                if guided:
                    uncond = flow(state, time, uncondition, mx.zeros(1))
            if guided:
                velocity = {
                    n: (1 + strength) * v.astype(mx.float32)
                    - strength * uncond[n].astype(mx.float32)
                    for n, v in velocity.items()
                }
            state = {
                n: x + (t1 - t0) * velocity[n].astype(mx.float32)
                for n, x in state.items()
            }
        else:
            if guided and hasattr(flow, "guided"):
                velocity, uncond = flow.guided(
                    state, grid, time, condition, uncondition
                )
            else:
                velocity = flow(state, grid, time, condition)
                if guided:
                    uncond = flow(state, grid, time, uncondition)
            if guided:
                velocity = (1 + strength) * velocity.astype(
                    mx.float32
                ) - strength * uncond.astype(mx.float32)
            state = state + (t1 - t0) * velocity.astype(mx.float32)
        mx.async_eval(state)
        yield i, state


def occupancy_grid(logits, prune=1):
    r = logits.shape[1]
    active = nonzero(logits[0, ..., 0] > 0)
    if not active.size:
        raise RuntimeError("The structure decoder produced no occupied voxels")
    coords = mx.stack(
        [mx.zeros_like(active), active // (r * r), active // r % r, active % r], axis=-1
    ).astype(mx.int32)
    grid = Grid(coords, r)
    if prune:
        if prune != 1:
            raise ValueError("Only the released one-voxel pruning radius is supported")
        surface = nonzero(mx.any(grid.neighbors() == coords.shape[0], axis=-1))
        coords = coords[surface]
    factor = 1
    if coords.shape[0] > 42000:
        xyz = coords[:, 1:].astype(mx.float32)
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        size = high - low + 1
        target_size = size / 2
        target_low = low + (size - target_size) / 2
        target_high = target_low + target_size - 1
        xyz = mx.round(
            (xyz - low) / mx.maximum(high - low, 1) * (target_size - 1) + target_low
        )
        xyz = mx.clip(
            xyz, target_low.astype(mx.int32), target_high.astype(mx.int32)
        ).astype(mx.int32)
        codes, _ = unique((xyz[:, 0] * r + xyz[:, 1]) * r + xyz[:, 2])
        coords = mx.stack(
            [mx.zeros_like(codes), codes // (r * r), codes // r % r, codes % r], axis=-1
        )
        if coords.shape[0] > 42000:
            raise RuntimeError(
                "Occupancy remains too large after upstream downsampling"
            )
        factor = 2
    return Grid(coords, r), factor


class Pipeline:
    """One model, one active stream; input backpressure is one request.

    GPU work is submitted with async_eval. Dynamic sparse topology requires
    scalar synchronization. Async iteration runs submissions on one worker,
    leaving the caller's event loop responsive; it never buffers whole videos.
    """

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        # Models loaded by other paths (e.g. mlx_vlm.utils.load_model) share too.
        model.share_backbones()

    @staticmethod
    def _conditions(flow, tokens, strength):
        """Project condition tokens once per request; CFG also needs the zero one."""
        prepared = [flow.prepare_condition(tokens)]
        if strength > 0:
            prepared.append(flow.prepare_condition(None))
        mx.async_eval(prepared)
        return prepared

    @classmethod
    def from_pretrained(cls, path):
        return cls(Model.from_pretrained(path))

    def _run(self, request, cancel):
        config, model = self.model.config, self.model
        formats = tuple(request.formats)
        if not formats or set(formats) - {"gaussian", "gaussian_4", "mesh"}:
            raise ValueError("formats must contain gaussian, gaussian_4, or mesh")
        ss_steps = config.ss_steps if request.ss_steps is None else request.ss_steps
        slat_steps = (
            config.slat_steps if request.slat_steps is None else request.slat_steps
        )
        schedule(ss_steps, config.ss_rescale_t)
        schedule(slat_steps, config.slat_rescale_t)
        pointmap = request.pointmap
        depth = getattr(model, "depth_model", None)
        if pointmap is None and request.estimate_depth and depth is not None:
            pointmap = estimate_pointmap(depth, request.image, config.depth_num_tokens)
            mx.async_eval(pointmap)
            yield Event(request.request_id, "depth", 1, 1, {"pointmap": pointmap})
        inputs, metadata = prepare_inputs(
            request.image, request.mask, pointmap, size=config.image_size
        )
        features = {} if model.shared_backbone else None
        condition = model.ss_condition_embedder(inputs, cache=features)
        conditions = self._conditions(model.ss_generator, condition, config.ss_guidance)
        yield Event(
            request.request_id,
            "conditioning",
            1,
            1,
            {"pointmap_conditioned": metadata["pointmap_conditioned"]},
        )
        keys = mx.random.split(mx.random.key(request.seed), len(LATENTS) + 1)
        state = {
            name: mx.random.normal(
                (1, config.latent_resolution**3 if name == "shape" else 1, channels),
                key=keys[i],
            )
            for i, (name, channels) in enumerate(LATENTS.items())
        }
        for step, state in sample(
            model.ss_generator,
            state,
            ss_steps,
            config.ss_rescale_t,
            config.ss_guidance,
            *conditions,
            cancel=cancel,
        ):
            yield Event(
                request.request_id, "structure", step, ss_steps, {"latents": state}
            )
        if cancel.is_set():
            return
        logits = model.ss_decoder(state["shape"])
        mx.async_eval(logits)
        grid, factor = occupancy_grid(logits, config.downsample_ss_dist)
        pose = decode_pose(state, metadata, factor)
        yield Event(
            request.request_id, "occupancy", 1, 1, {"coords": grid.coords, "pose": pose}
        )
        condition = model.slat_condition_embedder(inputs, cache=features)
        conditions = self._conditions(
            model.slat_generator, condition, config.slat_guidance
        )
        latent = mx.random.normal(
            (grid.coords.shape[0], config.latent_channels), key=keys[-1]
        )
        for step, latent in sample(
            model.slat_generator,
            latent,
            slat_steps,
            config.slat_rescale_t,
            config.slat_guidance,
            *conditions,
            grid=grid,
            cancel=cancel,
        ):
            yield Event(
                request.request_id,
                "latent",
                step,
                slat_steps,
                {"features": latent, "coords": grid.coords},
            )
        if cancel.is_set():
            return
        latent = latent * mx.array(config.slat_std) + mx.array(config.slat_mean)
        result = {
            "coords": grid.coords,
            "latents": latent,
            "pose": pose,
            "pointmap_conditioned": metadata["pointmap_conditioned"],
        }
        for index, kind in enumerate(formats, 1):
            if cancel.is_set():
                return
            if kind == "mesh":
                features, mesh_grid = model.slat_decoder_mesh(latent, grid)
                mx.async_eval(features)
                output = extract_mesh(features, mesh_grid)
            else:
                decoder = (
                    model.slat_decoder_gs
                    if kind == "gaussian"
                    else model.slat_decoder_gs_4
                )
                output = decoder(latent, grid)
            mx.async_eval(output)
            result[kind] = output
            yield Event(request.request_id, kind, index, len(formats), output)
        mx.async_eval(result)
        yield Event(request.request_id, "complete", 1, 1, result)

    def stream(self, request):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A stream is already using this pipeline")
        cancel = threading.Event()
        try:
            yield from self._run(request, cancel)
        finally:
            cancel.set()
            self._lock.release()

    def generate(self, image, mask=None, pointmap=None, **kwargs):
        for event in self.stream(Request(image, mask, pointmap, **kwargs)):
            if event.stage == "complete":
                mx.eval(event.data)
                return event.data

    async def astream(self, requests):
        """Consume an iterable or async iterable, yielding ordered Event objects.

        Cancellation stops further submissions and closes the active request.
        Already submitted GPU commands finish before worker stream cleanup.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A stream is already using this pipeline")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3d-mlx")
        loop = asyncio.get_running_loop()
        cancel = threading.Event()
        iterator = None

        def advance():
            return next(iterator, None)

        async def inputs():
            if hasattr(requests, "__aiter__"):
                async for item in requests:
                    yield item
            else:
                for item in requests:
                    yield item

        def cleanup():
            if iterator is not None:
                iterator.close()
            mx.synchronize()
            clear = getattr(mx, "clear_streams", None)
            if clear is not None:
                clear()

        try:
            async for request in inputs():
                iterator = self._run(request, cancel)
                while True:
                    event = await loop.run_in_executor(executor, advance)
                    if event is None:
                        break
                    yield event
                iterator = None
        finally:
            cancel.set()
            # Cleanup is queued behind any in-flight advance on the same worker.
            pending_cleanup = loop.run_in_executor(executor, cleanup)

            def release(_):
                executor.shutdown(wait=False)
                self._lock.release()

            pending_cleanup.add_done_callback(release)
            await asyncio.shield(pending_cleanup)
